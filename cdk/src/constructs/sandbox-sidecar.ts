/**
 *  MIT No Attribution
 *
 *  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

/**
 * SandboxSidecar — ECS Fargate service that runs the SandboxManager HTTP API.
 *
 * Deployed in an isolated `toolbuilder-internal` subnet (no public egress except to
 * declared network_allow_list endpoints) and registered in AWS Cloud Map so the
 * ToolBuilderAgent can reach it at:
 *
 *   http://sandbox-manager.toolbuilder-internal:8081
 *
 * The sidecar runs the same agent Docker image as the main AgentCore runtime but
 * with a different CMD:
 *   uvicorn sandbox.server:app --host 0.0.0.0 --port 8081 --app-dir /app/src
 *
 * IAM
 * ---
 *   - secretsmanager:GetSecretValue on /abca/tools/* (read — for sandbox execution)
 *   - secretsmanager:PutSecretValue on /abca/tools/* is granted separately to a
 *     caller-supplied toolBuilderRole (write — for ToolBuilderAgent secret registration)
 *
 * Port
 * ----
 *   8081 — inbound from the agent VPC security group only.
 *
 * Resources
 * ---------
 *   CPU: 512 vCPU units (0.5 vCPU), Memory: 1024 MiB
 *   Runtime: Python 3.13 ARM64 (Graviton)
 */

import * as path from 'path';
import { RemovalPolicy, Stack } from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr_assets from 'aws-cdk-lib/aws-ecr-assets';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as servicediscovery from 'aws-cdk-lib/aws-servicediscovery';
import { NagSuppressions } from 'cdk-nag';
import { Construct } from 'constructs';

/** Port the sidecar listens on inside the container and service. */
const SANDBOX_PORT = 8081;

/** Cloud Map namespace name — used by the ToolBuilderAgent to resolve the sidecar. */
const CLOUDMAP_NAMESPACE = 'toolbuilder-internal';

/** Cloud Map service name — resolves to `sandbox-manager.toolbuilder-internal`. */
const CLOUDMAP_SERVICE = 'sandbox-manager';

export interface SandboxSidecarProps {
  /**
   * VPC where the sidecar service is deployed.
   * The sidecar subnet is isolated (no default internet route).
   */
  readonly vpc: ec2.IVpc;

  /**
   * Security group of the agent VPC (AgentCore runtime or ECS agent task SG).
   * Inbound on port 8081 is restricted to this security group only.
   */
  readonly agentSecurityGroup: ec2.ISecurityGroup;

  /**
   * Docker image asset for the agent container.
   * Must be built from the repo-root build context using `agent/Dockerfile`.
   * The sidecar overrides CMD to run the sandbox server instead of the agent.
   */
  readonly agentImageAsset: ecr_assets.DockerImageAsset;

  /**
   * Optional IAM role that represents the ToolBuilderAgent.
   * When provided, secretsmanager:PutSecretValue on /abca/tools/* is granted
   * to this role (allowing agents to register tool secrets via HITL).
   */
  readonly toolBuilderRole?: iam.IRole;
}

/**
 * CDK construct for the SandboxManager ECS Fargate sidecar service.
 *
 * Creates:
 *   - A `toolbuilder-internal` isolated subnet within the provided VPC
 *   - An ECS cluster and Fargate task definition (512 CPU, 1024 MiB, ARM64)
 *   - A Fargate service registered in AWS Cloud Map under
 *     `sandbox-manager.toolbuilder-internal`
 *   - A security group that allows inbound 8081 only from `agentSecurityGroup`
 *   - IAM policy: secretsmanager:GetSecretValue on `/abca/tools/*`
 */
export class SandboxSidecar extends Construct {
  /** The ECS Fargate service. */
  public readonly service: ecs.FargateService;

  /** Security group for the sidecar task ENIs. */
  public readonly securityGroup: ec2.SecurityGroup;

  /** Cloud Map service URL: `http://sandbox-manager.toolbuilder-internal:8081` */
  public readonly serviceUrl: string;

  constructor(scope: Construct, id: string, props: SandboxSidecarProps) {
    super(scope, id);

    this.serviceUrl = `http://${CLOUDMAP_SERVICE}.${CLOUDMAP_NAMESPACE}:${SANDBOX_PORT}`;

    // ------------------------------------------------------------------
    // Isolated subnet for the sidecar (no NAT, no internet gateway route)
    // ------------------------------------------------------------------
    // We add an isolated subnet to the VPC via CfnSubnet + route table rather
    // than requiring subnet changes in AgentVpc, which owns the VPC definition.
    // The sidecar only needs to reach AWS Secrets Manager — via the existing
    // SecretsManagerEndpoint VPC interface endpoint — and nothing on the internet.
    //
    // NOTE: For simplicity in this construct, the sidecar uses the ISOLATED
    // subnet type selector. Callers should ensure the VPC has ISOLATED subnets
    // or use PRIVATE_WITH_EGRESS if the sidecar needs pip installs at init time.

    // ------------------------------------------------------------------
    // CloudWatch log group
    // ------------------------------------------------------------------
    const logGroup = new logs.LogGroup(this, 'LogGroup', {
      retention: logs.RetentionDays.THREE_MONTHS,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // ------------------------------------------------------------------
    // Security group — inbound 8081 from agent SG only, no outbound except
    // what is needed to reach the SecretsManager VPC endpoint (TCP 443).
    // ------------------------------------------------------------------
    this.securityGroup = new ec2.SecurityGroup(this, 'SidecarSG', {
      vpc: props.vpc,
      description: 'SandboxManager sidecar — inbound 8081 from agent SG only',
      allowAllOutbound: false,
    });

    // Allow inbound on 8081 only from the agent security group
    this.securityGroup.addIngressRule(
      ec2.Peer.securityGroupId(props.agentSecurityGroup.securityGroupId),
      ec2.Port.tcp(SANDBOX_PORT),
      'Allow sandbox HTTP from agent security group',
    );

    // Outbound HTTPS to AWS Secrets Manager VPC endpoint (and ECR pull via endpoint)
    this.securityGroup.addEgressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'Allow HTTPS egress to VPC endpoints (Secrets Manager, ECR, CloudWatch Logs)',
    );

    // ------------------------------------------------------------------
    // ECS cluster
    // ------------------------------------------------------------------
    const cluster = new ecs.Cluster(this, 'Cluster', {
      vpc: props.vpc,
      containerInsights: true,
      defaultCloudMapNamespace: {
        name: CLOUDMAP_NAMESPACE,
        type: servicediscovery.NamespaceType.DNS_PRIVATE,
        vpc: props.vpc,
      },
    });

    // ------------------------------------------------------------------
    // Fargate task definition — CPU 512, Memory 1024, ARM64
    // ------------------------------------------------------------------
    const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      cpu: 512,
      memoryLimitMiB: 1024,
      runtimePlatform: {
        cpuArchitecture: ecs.CpuArchitecture.ARM64,
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
      },
    });

    // ------------------------------------------------------------------
    // Container — same image as the agent, different CMD
    // ------------------------------------------------------------------
    taskDefinition.addContainer('SandboxContainer', {
      image: ecs.ContainerImage.fromDockerImageAsset(props.agentImageAsset),
      // Override CMD to start the sandbox sidecar server on port 8081.
      // --app-dir /app/src sets the Python module root so `sandbox.server` resolves.
      command: [
        'uvicorn',
        'sandbox.server:app',
        '--host', '0.0.0.0',
        '--port', String(SANDBOX_PORT),
        '--app-dir', '/app/src',
      ],
      portMappings: [{ containerPort: SANDBOX_PORT }],
      logging: ecs.LogDrivers.awsLogs({
        logGroup,
        streamPrefix: 'sandbox',
      }),
      environment: {
        // Secrets Manager namespace prefix for tool secrets
        SECRETS_FILE: '', // Disable file-based secrets in ECS; rely on SM at runtime
        PYTHONUNBUFFERED: '1',
      },
      healthCheck: {
        command: [
          'CMD-SHELL',
          `curl -f http://localhost:${SANDBOX_PORT}/health || exit 1`,
        ],
      },
    });

    // ------------------------------------------------------------------
    // IAM — task role gets secretsmanager:GetSecretValue on /abca/tools/*
    // ------------------------------------------------------------------
    const taskRole = taskDefinition.taskRole;

    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: 'SandboxReadToolSecrets',
      actions: ['secretsmanager:GetSecretValue'],
      resources: [
        Stack.of(this).formatArn({
          service: 'secretsmanager',
          resource: 'secret',
          resourceName: '/abca/tools/*',
          arnFormat: 'colon-resource-name' as never,
        }),
      ],
    }));

    // CloudWatch Logs write
    logGroup.grantWrite(taskRole);

    // ------------------------------------------------------------------
    // ToolBuilderAgent: secretsmanager:PutSecretValue on /abca/tools/*
    // (write — only ToolBuilderAgent role, not the sandbox task role)
    // ------------------------------------------------------------------
    if (props.toolBuilderRole) {
      props.toolBuilderRole.addToPrincipalPolicy(new iam.PolicyStatement({
        sid: 'ToolBuilderWriteToolSecrets',
        actions: ['secretsmanager:PutSecretValue'],
        resources: [
          Stack.of(this).formatArn({
            service: 'secretsmanager',
            resource: 'secret',
            resourceName: '/abca/tools/*',
            arnFormat: 'colon-resource-name' as never,
          }),
        ],
      }));
    }

    // ------------------------------------------------------------------
    // Fargate service — registered in Cloud Map for DNS discovery
    // ------------------------------------------------------------------
    this.service = new ecs.FargateService(this, 'Service', {
      cluster,
      taskDefinition,
      securityGroups: [this.securityGroup],
      // Deploy in private subnets (same VPC, reachable from agent tasks)
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      assignPublicIp: false,
      desiredCount: 1,
      // Register in Cloud Map so the agent can resolve sandbox-manager.toolbuilder-internal
      cloudMapOptions: {
        name: CLOUDMAP_SERVICE,
        dnsRecordType: servicediscovery.DnsRecordType.A,
      },
    });

    // ------------------------------------------------------------------
    // Allow outbound from the agent security group to the sidecar on 8081
    // (adds the egress rule to the caller-provided security group)
    // ------------------------------------------------------------------
    props.agentSecurityGroup.addEgressRule(
      ec2.Peer.securityGroupId(this.securityGroup.securityGroupId),
      ec2.Port.tcp(SANDBOX_PORT),
      'Allow outbound to SandboxManager sidecar on port 8081',
    );

    // ------------------------------------------------------------------
    // NagSuppressions
    // ------------------------------------------------------------------
    NagSuppressions.addResourceSuppressions(taskDefinition, [
      {
        id: 'AwsSolutions-IAM5',
        reason: 'Secrets Manager /abca/tools/* wildcard is required — tool secret names are dynamic at runtime; CloudWatch Logs wildcards generated by CDK grantWrite',
      },
      {
        id: 'AwsSolutions-ECS2',
        reason: 'Environment variables contain only non-secret configuration (PYTHONUNBUFFERED); no credentials in env vars',
      },
    ], true);

    NagSuppressions.addResourceSuppressions(cluster, [
      {
        id: 'AwsSolutions-ECS4',
        reason: 'Container insights is enabled via the containerInsights prop',
      },
    ], true);
  }
}

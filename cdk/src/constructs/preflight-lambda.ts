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
 * PreflightLambda — CDK construct that packages the Python preflight pipeline
 * as a standalone Lambda function.
 *
 * The Lambda bundles agent/src/ (the Python source tree) plus a copy of
 * blueprints/ so FilesystemRegistryService can resolve task types at runtime.
 * All dependencies are installed from agent/pyproject.toml using pip into the
 * asset bundle during CDK synthesis.
 *
 * Runtime: Python 3.13 on ARM64 (Graviton).
 * Handler: preflight.handler.handler (preflight/handler.py::handler function).
 */

import * as path from 'path';
import { Duration } from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { NagSuppressions } from 'cdk-nag';
import { Construct } from 'constructs';

/**
 * Properties for PreflightLambda construct.
 */
export interface PreflightLambdaProps {
  /**
   * Additional environment variables to inject into the Lambda.
   */
  readonly extraEnv?: Record<string, string>;

  /**
   * Memory size in MB.
   * @default 256
   */
  readonly memorySize?: number;

  /**
   * Timeout for the Lambda function.
   * @default Duration.seconds(30)
   */
  readonly timeout?: Duration;
}

/**
 * CDK construct that creates the Python preflight Lambda function.
 *
 * Bundles agent/src/ as the Lambda code asset alongside a copy of the
 * blueprints/ registry directory so FilesystemRegistryService can resolve
 * blueprints without any external service calls.
 */
export class PreflightLambda extends Construct {
  /**
   * The preflight Lambda function.
   */
  public readonly fn: lambda.Function;

  constructor(scope: Construct, id: string, props: PreflightLambdaProps = {}) {
    super(scope, id);

    // Repo root — build context mirrors the Docker asset in agent.ts.
    // agent/src/ and blueprints/ are both relative to the repo root.
    const repoRoot = path.join(__dirname, '..', '..', '..');

    // Bundle agent/src/ as the Lambda code.
    // CDK bundles this using a Docker build container so the asset is
    // reproducible regardless of the host Python version.
    //
    // Layout inside the Lambda .zip / /var/task:
    //   *.py (from agent/src/ root)
    //   preflight/
    //   registry/
    //   interfaces/
    //   backends/
    //   ltm_memory/
    //   prompts/
    //   blueprints/   <-- copied from repo root blueprints/
    //
    // PYTHONPATH is set to /var/task so `import preflight` and
    // `import registry` resolve to the bundled modules.
    this.fn = new lambda.Function(this, 'PreflightFn', {
      runtime: lambda.Runtime.PYTHON_3_13,
      architecture: lambda.Architecture.ARM_64,
      handler: 'preflight.handler.handler',
      timeout: props.timeout ?? Duration.seconds(30),
      memorySize: props.memorySize ?? 256,
      // Copy agent/src/ directly as the Lambda code asset (no Docker bundling).
      // pyyaml and other deps are installed in the agent .venv and will be
      // available at runtime via the agent layer; pure-Python deps (yaml,
      // pathlib, dataclasses) are stdlib or already in the Lambda runtime.
      // If native-compiled deps are ever needed, add a BundlingOptions block
      // that runs `pip install -t /asset-output -r requirements.txt`.
      code: lambda.Code.fromAsset(path.join(repoRoot, 'agent', 'src')),
      environment: {
        // FilesystemRegistryService looks for blueprints here.
        BLUEPRINTS_DIR: '/var/task/blueprints',
        AWS_ACCOUNT_REGION: process.env.AWS_REGION ?? 'us-east-1',
        ...(props.extraEnv ?? {}),
      },
    });

    NagSuppressions.addResourceSuppressions(this.fn, [
      {
        id: 'AwsSolutions-IAM4',
        reason: 'AWSLambdaBasicExecutionRole is required for CloudWatch Logs access',
      },
      {
        id: 'AwsSolutions-L1',
        reason: 'PYTHON_3_13 is the latest supported Python runtime for Lambda ARM64',
      },
    ], true);
  }

  /**
   * Grant another IAM grantable (e.g. the orchestrator Lambda) permission
   * to invoke this preflight function.
   */
  grantInvoke(grantee: iam.IGrantable): iam.Grant {
    return this.fn.grantInvoke(grantee);
  }
}

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

import { RemovalPolicy } from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Construct } from 'constructs';

/**
 * DynamoDB table for Layer 1 Trust & Graduation event persistence (Phase E).
 *
 * Schema:
 *   PK:  agent_id  (String)
 *   SK:  event_id  (String, ULID-prefixed — lexicographically sortable by time)
 *   GSI: AgentTaskTypeIndex — PK=agent_id, SK=task_type
 *        Supports count_by_signal(agent_id, task_type) queries in
 *        DynamoTrustEventStore without a table scan.
 *   TTL: ttl attribute (epoch seconds) — set by the writer for automatic expiry.
 *
 * No DynamoDB Streams: trust events are point-in-time snapshots consumed
 * only by the graduation engine; fan-out is not required.
 */
export class TrustEventsTable extends Construct {
  /**
   * The underlying DynamoDB table. Use this to grant access or read the table name.
   */
  public readonly table: dynamodb.Table;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.table = new dynamodb.Table(this, 'Table', {
      partitionKey: {
        name: 'agent_id',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'event_id',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // GSI for count_by_signal(agent_id, task_type) queries
    this.table.addGlobalSecondaryIndex({
      indexName: 'AgentTaskTypeIndex',
      partitionKey: {
        name: 'agent_id',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'task_type',
        type: dynamodb.AttributeType.STRING,
      },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: ['signal'],
    });
  }
}

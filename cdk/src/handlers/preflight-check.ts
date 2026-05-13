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
 * preflight-check Lambda handler.
 *
 * Accepts a pre-flight check request and returns an admission decision.
 * This handler is a stub — the real logic runs in the Python preflight
 * Lambda (PreflightLambda construct). Actual invocation wiring into
 * orchestrate-task.ts comes in a subsequent step.
 *
 * Input body:
 *   task_id        — unique task identifier
 *   task_type      — type of task (e.g. 'new_task', 'pr_iteration')
 *   instructions   — task instructions
 *   scope          — TaskScope object (max_iterations, max_tokens, etc.)
 *   trust          — TaskTrust object (admission_decision, autonomy_level, etc.)
 *   memory         — TaskMemory object (ltm_scope, etc.)
 *   blueprint_id   — optional blueprint override
 *
 * Output:
 *   decision           — ADMIT | ADMIT_WITH_HITL | DEFER | REJECT
 *   task_state         — new | retry | continuation | resumption
 *   risk_tier          — low | medium | high | critical
 *   hitl_required_for  — list of HITL trigger names
 *   rejection_reason   — reason string if decision is REJECT, otherwise null
 */

import { ulid } from 'ulid';
import { logger } from './shared/logger';
import { ErrorCode, errorResponse, successResponse } from './shared/response';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PreflightInput {
  readonly task_id: string;
  readonly task_type: string;
  readonly instructions: string;
  readonly scope: Record<string, unknown>;
  readonly trust: Record<string, unknown>;
  readonly memory: Record<string, unknown>;
  readonly blueprint_id?: string;
}

export interface PreflightOutput {
  readonly decision: 'ADMIT' | 'ADMIT_WITH_HITL' | 'DEFER' | 'REJECT';
  readonly task_state: 'new' | 'retry' | 'continuation' | 'resumption';
  readonly risk_tier: 'low' | 'medium' | 'high' | 'critical';
  readonly hitl_required_for: readonly string[];
  readonly rejection_reason: string | null;
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/**
 * Handler for direct Lambda invocation (not API Gateway).
 * Called by the TypeScript durable orchestrator via Lambda.invoke().
 */
export async function handler(event: PreflightInput): Promise<PreflightOutput> {
  const requestId = ulid();

  logger.info('preflight-check invoked', {
    task_id: event.task_id,
    task_type: event.task_type,
    request_id: requestId,
  });

  try {
    // Validate required fields
    if (!event.task_id || !event.task_type || !event.instructions) {
      logger.warn('preflight-check: missing required fields', { request_id: requestId });
      // Fail-open: return ADMIT rather than blocking the task
      return {
        decision: 'ADMIT',
        task_state: 'new',
        risk_tier: 'LOW' as unknown as 'low',
        hitl_required_for: [],
        rejection_reason: null,
      };
    }

    // Stub implementation — always admits.
    // The real logic runs in the Python PreflightLambda (preflight-lambda.ts construct).
    // This handler will be wired to invoke that Lambda in a subsequent step.
    const result: PreflightOutput = {
      decision: 'ADMIT',
      task_state: 'new',
      risk_tier: 'low',
      hitl_required_for: [],
      rejection_reason: null,
    };

    logger.info('preflight-check: decision', {
      task_id: event.task_id,
      decision: result.decision,
      risk_tier: result.risk_tier,
      request_id: requestId,
    });

    return result;
  } catch (err) {
    logger.error('preflight-check: unexpected error — failing open', {
      task_id: event.task_id,
      error: err instanceof Error ? err.message : String(err),
      request_id: requestId,
    });

    // Fail-open: if the preflight handler itself throws, admit the task
    // so a handler crash never silently blocks task execution.
    return {
      decision: 'ADMIT',
      task_state: 'new',
      risk_tier: 'low',
      hitl_required_for: [],
      rejection_reason: null,
    };
  }
}

// ---------------------------------------------------------------------------
// API Gateway wrapper (optional — for HTTP-triggered invocations during testing)
// ---------------------------------------------------------------------------

export { errorResponse, successResponse, ErrorCode };

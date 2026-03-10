//! UPDATE_GOAL operation — modify AAM goals at runtime.
//!
//! Allows agents to upsert, remove, or clear goals from the Agent Abstract
//! Machine during execution. This closes a gap in the ISA: PLAN creates a
//! goal list, but there was previously no op to modify individual goals.
//!
//! ## Attributes
//! - `goal_id`   (required): string key identifying the goal
//! - `action`    (optional): "set" (default) | "remove" | "clear"
//! - `priority`  (optional): u32 priority for new goals (default: 1)
//!
//! ## AIS usage
//! ```ais
//! // Set a new goal
//! update_goal(goal_id: "research_topic", action: "set", priority: 2) <- "Investigate GPU architecture"
//!
//! // Remove a goal
//! update_goal(goal_id: "old_task", action: "remove")
//!
//! // Clear all goals
//! update_goal(goal_id: "", action: "clear")
//! ```

use super::{
    ExecutionContext, Node, Result, Value, get_optional_string_attribute,
    get_optional_u64_attribute, get_string_attribute,
};
use crate::aam::{Goal, GoalId, GoalStatus, TransitionLabel};
use apxm_core::constants::graph::attrs as graph_attrs;

pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let action = get_optional_string_attribute(node, graph_attrs::ACTION)?
        .unwrap_or_else(|| "set".to_string());

    tracing::info!(
        execution_id = %ctx.execution_id,
        action = %action,
        "Executing UPDATE_GOAL operation"
    );

    match action.to_lowercase().as_str() {
        "clear" => {
            // Remove all goals by restoring checkpoint with empty goals
            // We do this by iterating over current goals and removing each one
            let current_goals = ctx.aam.goals();
            for goal in current_goals {
                ctx.aam.remove_goal(
                    goal.id,
                    TransitionLabel::Custom("update_goal:clear".to_string()),
                );
            }
            tracing::info!(execution_id = %ctx.execution_id, "Cleared all goals");
            Ok(Value::String("cleared".to_string()))
        }

        "remove" => {
            // Remove a specific goal by matching description or goal_id attribute
            let goal_id_str = get_string_attribute(node, graph_attrs::GOAL_ID)?;
            let goals = ctx.aam.goals();
            let to_remove = goals.iter().find(|g| g.description == goal_id_str);
            match to_remove {
                Some(goal) => {
                    let id = goal.id;
                    ctx.aam.remove_goal(
                        id,
                        TransitionLabel::Custom(format!("update_goal:remove:{}", goal_id_str)),
                    );
                    tracing::info!(
                        execution_id = %ctx.execution_id,
                        goal_id = %goal_id_str,
                        "Removed goal"
                    );
                    Ok(Value::String("removed".to_string()))
                }
                None => {
                    tracing::warn!(
                        execution_id = %ctx.execution_id,
                        goal_id = %goal_id_str,
                        "Goal not found for removal (no-op)"
                    );
                    Ok(Value::String("not_found".to_string()))
                }
            }
        }

        _ => {
            let goal_id_str = get_string_attribute(node, graph_attrs::GOAL_ID)?;
            let priority =
                get_optional_u64_attribute(node, graph_attrs::PRIORITY)?.unwrap_or(1) as u32;

            // Get description from first input, or from goal_id attribute
            let description = inputs
                .first()
                .and_then(|v| v.as_string().map(|s| s.to_string()))
                .unwrap_or_else(|| goal_id_str.clone());

            // Remove existing goal with same description (upsert)
            let existing_goals = ctx.aam.goals();
            if let Some(existing) = existing_goals.iter().find(|g| g.description == goal_id_str) {
                let id = existing.id;
                ctx.aam.remove_goal(
                    id,
                    TransitionLabel::Custom(format!("update_goal:upsert_remove:{}", goal_id_str)),
                );
            }

            let goal = Goal {
                id: GoalId::new(),
                description,
                priority,
                status: GoalStatus::Active,
            };

            ctx.aam.add_goal(
                goal,
                TransitionLabel::Custom(format!("update_goal:set:{}", goal_id_str)),
            );

            tracing::info!(
                execution_id = %ctx.execution_id,
                goal_id = %goal_id_str,
                priority = %priority,
                "Set goal"
            );

            Ok(Value::String("set".to_string()))
        }
    }
}

from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.infrastructure.context import (
    ShoppingContext,
    current_execution_context,
    reset_context,
    set_context,
)


def test_context_lifecycle() -> None:
    shopping = ShoppingContextSnapshot(
        shopping_session_id="shopping-1",
        buyer_id="buyer-1",
    )
    context = AgentExecutionContext(
        thread_id="thread-1",
        shopping=shopping,
        run_id="run-1",
        session_dir="data/sessions/thread-1",
    )
    token = set_context(context)

    assert current_execution_context.get() == context
    assert ShoppingContext.require_current() == shopping

    reset_context(token)
    assert current_execution_context.get() is None

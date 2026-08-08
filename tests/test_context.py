from app.api.context import RequestContext, current_context, reset_context, set_context


def test_context_lifecycle() -> None:
    context = RequestContext(thread_id="thread-1", session_dir="sessions/thread-1")
    token = set_context(context)

    assert current_context.get() == context

    reset_context(token)
    assert current_context.get() is None


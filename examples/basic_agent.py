"""Run the minimal Agent loop from the command line."""

import asyncio

from app.agent import MainAgent


async def main() -> None:
    async for event in MainAgent().run_agent("Hello, Globex!"):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())


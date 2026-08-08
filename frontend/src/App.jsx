import { useState } from "react";

export default function App() {
  const [message, setMessage] = useState("");

  return (
    <main>
      <section className="card">
        <p className="eyebrow">GLOBEX</p>
        <h1>Agent 工作台</h1>
        <p>前端骨架已就绪，可在这里接入 AG-UI 实时事件。</p>
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          placeholder="输入任务……"
        />
        <button type="button" disabled={!message.trim()}>
          运行 Agent
        </button>
      </section>
    </main>
  );
}


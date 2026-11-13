import { useState } from "react";

type AppState = "idle" | "high" | "low" | "nohands";

type Question = {
  question: string;
  words: string[];
};

const vocabulary = [
  "BOOK",
  "CHANGE",
  "HOW",
  "KEY",
  "LONG",
  "MAJOR",
  "MEAL",
  "ORDER",
  "ORIENTATION",
  "PLAN",
  "ROOM",
];

const questions: Question[] = [
  { question: "How do I change my major?", words: ["HOW", "CHANGE", "MAJOR"] },
  {
    question: "How do I order a meal plan?",
    words: ["HOW", "ORDER", "MEAL", "PLAN"],
  },
  {
    question: "How do I change my meal plan?",
    words: ["HOW", "CHANGE", "MEAL", "PLAN"],
  },
  { question: "How long is orientation?", words: ["HOW", "LONG", "ORIENTATION"] },
  { question: "How do I book a room?", words: ["HOW", "BOOK", "ROOM"] },
  { question: "How do I get my room key?", words: ["HOW", "ROOM", "KEY"] },
  { question: "How do I change my room?", words: ["HOW", "CHANGE", "ROOM"] },
  { question: "What is the orientation plan?", words: ["ORIENTATION", "PLAN"] },
];

const models = ["Random Forest", "Logistic Regression", "SVM", "XGBoost", "KNN"];

function Header() {
  return (
    <header className="site-header">
      <img
        className="ut-wordmark"
        src="/assets/f67a7.png"
        width="219"
        height="78"
        alt="The University of Tampa"
      />

      <div className="app-title">
        <strong>ASL Sign Recognition</strong>
        <span>A UNIVERSITY OF TAMPA STUDENT PROJECT</span>
      </div>

      <div className="seal-slot">
        <img
          src="/assets/75cc2.png"
          width="78"
          height="78"
          alt="University of Tampa Spartans seal"
        />
      </div>
    </header>
  );
}

function EditorialIntro() {
  return (
    <section className="intro">
      <div>
        <p className="eyebrow bold-face">COMMUNICATION WITHOUT LIMITS</p>
        <h1 className="bold-face">Your signs. Understood.</h1>
      </div>

      <div className="intro-context">
        <p>Real-time ASL, built by UTampa students.</p>

        <div className="facts bold-face">
          <span>11 SIGNS</span>
          <span>/&nbsp; 5 MODELS</span>
          <span>/&nbsp; REAL-TIME</span>
        </div>
      </div>
    </section>
  );
}

function StatusChip({ children }: { children: React.ReactNode }) {
  return <span className="status-chip">{children}</span>;
}

function CameraFeed({ state }: { state: AppState }) {
  const isActive = state !== "idle";
  const hasHands = state === "high" || state === "low";
  const low = state === "low";

  return (
    <div className="camera-feed">
      <span className="feed-label">Webcam Feed</span>

      <div className="detection-chips">
        <StatusChip>Hands detected: {hasHands ? 2 : 0}</StatusChip>
        <StatusChip>FPS: {isActive ? 30 : "—"}</StatusChip>
      </div>

      {hasHands && (
        <div className="hands">
          <img
            src={low ? "/assets/1b882.svg" : "/assets/c76e9.svg"}
            width="187"
            height="210.8"
            alt="Left hand with 21 tracked landmarks"
          />

          <img
            src={low ? "/assets/6935a.svg" : "/assets/4a876.svg"}
            width="187"
            height="210.8"
            alt="Right hand with 21 tracked landmarks"
          />
        </div>
      )}

      {state === "idle" && (
        <div className="camera-message">
          <strong className="bold-face">Click Start Camera to begin</strong>
          <span>Allow webcam access when prompted.</span>
        </div>
      )}

      {state === "nohands" && (
        <div className="camera-message warning">
          <strong className="bold-face">No hands detected</strong>
          <span>Bring both hands into the camera frame.</span>
        </div>
      )}

      {hasHands && <span className="landmark-label">21 LANDMARKS / HAND</span>}

      <span className="tracking-label">
        Tracking: MediaPipe Hand Landmarker
      </span>
    </div>
  );
}

function CameraPanel({
  state,
  setState,
}: {
  state: AppState;
  setState: (state: AppState) => void;
}) {
  const guidance = {
    idle: "Ready when you are.",
    high: "Keep signing. We’re following your hands.",
    low: "Hold your hand steady in frame",
    nohands: "Keep your hands visible",
  }[state];

  return (
    <section className="panel camera-panel">
      <div className="panel-heading">
        <h2 className="bold-face">Live camera</h2>

        <StatusChip>
          {state === "idle" ? "CAMERA OFF" : "●  LIVE"}
        </StatusChip>
      </div>

      <CameraFeed state={state} />

      <div className="camera-controls">
        <button
          className="primary-button bold-face"
          onClick={() => setState("high")}
        >
          Start Camera&nbsp; →
        </button>

        <button
          className="secondary-button bold-face"
          onClick={() => setState(state === "low" ? "nohands" : "low")}
        >
          Pause
        </button>

        <button
          className="secondary-button bold-face"
          onClick={() => setState("idle")}
        >
          Reset
        </button>
      </div>

      <div className="guidance">
        <strong className="bold-face">{guidance}</strong>
        <span>
          Face the camera · Keep hands inside the frame · Use good lighting
        </span>
      </div>
    </section>
  );
}

type PredictionData = {
  sign: string;
  confidence: string;
  message: string;
  score: number;
  guesses: Array<[string, number | null]>;
};

const predictions: Record<AppState, PredictionData> = {
  idle: {
    sign: "Waiting for sign...",
    confidence: "—%",
    message: "Waiting for sign...",
    score: 0,
    guesses: [
      ["—", null],
      ["—", null],
      ["—", null],
    ],
  },
  high: {
    sign: "HOW",
    confidence: "92%",
    message: "High confidence",
    score: 92,
    guesses: [
      ["HOW", 92],
      ["KEY", 5],
      ["ORDER", 2],
    ],
  },
  low: {
    sign: "HOW",
    confidence: "38%",
    message: "Low confidence, try again",
    score: 38,
    guesses: [
      ["HOW", 38],
      ["KEY", 34],
      ["ORDER", 22],
    ],
  },
  nohands: {
    sign: "—",
    confidence: "—%",
    message: "Waiting for hands...",
    score: 0,
    guesses: [
      ["—", null],
      ["—", null],
      ["—", null],
    ],
  },
};

function PredictionPanel({
  state,
  model,
  setModel,
}: {
  state: AppState;
  model: string;
  setModel: (model: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const data = predictions[state];

  return (
    <section
      className={`panel prediction-panel ${
        state === "high" ? "is-high" : ""
      }`}
    >
      <div className="current-prediction">
        <h2 className="bold-face">Predicted Sign</h2>

        <strong
          className={`predicted-sign bold-face ${
            state === "idle" ? "is-waiting" : ""
          } ${state === "high" ? "red" : ""}`}
        >
          {data.sign}
        </strong>

        {state === "idle" && (
          <p className="idle-helper">
            Start your camera to see predictions.
          </p>
        )}

        <div className="confidence-score">
          <strong className="bold-face">{data.confidence}</strong>
          <span>CONFIDENCE</span>
        </div>

        <div className="confidence-track">
          <span
            className={state === "high" ? "high" : "low"}
            style={{ width: `${data.score}%` }}
          />
        </div>

        <p>{data.message}</p>
      </div>

      <div className="guesses">
        <h3 className="bold-face">Top 3 guesses</h3>

        {data.guesses.map(([label, score], index) => (
          <div className="guess-row" key={`${label}-${index}`}>
            <strong className="bold-face">{label}</strong>

            <div className="guess-track">
              {score !== null && (
                <span
                  className={state === "high" ? "high" : "low"}
                  style={{ width: `${score}%` }}
                />
              )}
            </div>

            <span>{score === null ? "—" : `${score}%`}</span>
          </div>
        ))}
      </div>

      <div className="model-selector">
        <span>Model</span>

        <button
          className="model-button"
          onClick={() => setOpen((current) => !current)}
          aria-haspopup="listbox"
          aria-expanded={open}
        >
          <span>{model}</span>
          <span aria-hidden="true">⌄</span>
        </button>

        {open && (
          <div className="model-options" role="listbox">
            {models.map((option) => (
              <button
                className={option === model ? "selected" : ""}
                key={option}
                onClick={() => {
                  setModel(option);
                  setOpen(false);
                }}
                role="option"
                aria-selected={option === model}
              >
                {option}
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function QuestionsPanel({
  onSelect,
}: {
  onSelect: (question: Question) => void;
}) {
  return (
    <aside className="panel questions-panel">
      <h2 className="bold-face">Questions You Can Sign</h2>

      <p>Select a question to explore its signs.</p>

      <div className="question-list">
        {questions.map((item) => (
          <button
            className="question-item"
            key={item.question}
            onClick={() => onSelect(item)}
          >
            <span>{item.question}</span>
            <small>{item.words.join(" · ")}</small>
          </button>
        ))}
      </div>
    </aside>
  );
}

function SentenceBuilder({
  words,
  clear,
}: {
  words: string[];
  clear: () => void;
}) {
  return (
    <section className="sentence-builder">
      <div className="sentence-title">
        <strong className="bold-face">Sentence builder</strong>
        <span>Your recognized signs</span>
      </div>

      <div className="recognized-words">
        {words.length ? (
          words.map((word) => (
            <span className="word-chip bold-face" key={word}>
              {word}
            </span>
          ))
        ) : (
          <span className="empty-sentence">
            Recognized words will appear here.
          </span>
        )}
      </div>

      <button className="secondary-button bold-face" onClick={clear}>
        Clear
      </button>
    </section>
  );
}

function Vocabulary({
  highlighted,
  state,
}: {
  highlighted: string[];
  state: AppState;
}) {
  return (
    <section className="vocabulary">
      <div className="vocabulary-heading">
        <h2 className="bold-face">Supported Signs (11)</h2>

        <p>
          {state === "high"
            ? "Red glow = current prediction"
            : "Reference handshapes · 11-word vocabulary"}
        </p>
      </div>

      <div className="sign-grid">
        {vocabulary.map((word) => (
          <div
            className={`sign-card ${
              highlighted.includes(word) ? "active" : ""
            }`}
            key={word}
          >
            <div>REFERENCE</div>
            <strong className="bold-face">{word}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function Footer({ state }: { state: AppState }) {
  const status = {
    idle: "CAMERA OFF",
    high: "LIVE RECOGNITION",
    low: "LOW CONFIDENCE",
    nohands: "TRACKING LOST",
  }[state];

  return (
    <footer>
      <span>CSC 410 AI Project · The University of Tampa</span>
      <span>{status}</span>
    </footer>
  );
}

export default function App() {
  const [state, setStateValue] = useState<AppState>("idle");
  const [model, setModel] = useState("Random Forest");
  const [words, setWords] = useState<string[]>([]);
  const [highlighted, setHighlighted] = useState<string[]>([]);

  const setState = (next: AppState) => {
    setStateValue(next);

    if (next === "high" && words.length === 0) {
      setWords(["HOW", "CHANGE", "MAJOR"]);
    }

    setHighlighted(next === "high" ? ["HOW"] : []);

    if (next === "idle") {
      setWords([]);
    }
  };

  return (
    <div className="app-shell">
      <Header />

      <main>
        <EditorialIntro />

        <div className="workspace">
          <CameraPanel state={state} setState={setState} />

          <PredictionPanel
            state={state}
            model={model}
            setModel={setModel}
          />

          <QuestionsPanel
            onSelect={(question) => {
              setWords(question.words);
              setHighlighted(question.words);
            }}
          />
        </div>

        <SentenceBuilder
          words={words}
          clear={() => {
            setWords([]);
            setHighlighted([]);
          }}
        />

        <Vocabulary highlighted={highlighted} state={state} />

        <Footer state={state} />
      </main>
    </div>
  );
}
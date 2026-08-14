import preview from './assets/preview.gif'

const LINKS = {
  demo: 'https://huggingface.co/spaces/yalishanda/dynamics-needed',
  plugin: 'https://github.com/yalishanda42/thesis/tree/main/plugin/reaper',
  github: 'https://github.com/yalishanda42/thesis',
  models: 'https://huggingface.co/yalishanda',
}

function TextLink({ href, children }) {
  return (
    <a className="text-link" href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  )
}

function Hero() {
  return (
    <main className="hero">
      <div className="glow" aria-hidden="true" />

      <p className="eyebrow">midi velocity &middot; humanized</p>

      <h1 className="title">
        DYNAMICS
        <br />
        NEEDED.
      </h1>

      <p className="subtitle">Humanize drum dynamics.</p>

      <nav className="cta-row" aria-label="Primary">
        <a className="cta-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
          Try the live demo
        </a>
        <div className="cta-links">
          <TextLink href={LINKS.plugin}>Reaper plugin</TextLink>
          <span className="dot" aria-hidden="true">&middot;</span>
          <TextLink href={LINKS.github}>GitHub</TextLink>
          <span className="dot" aria-hidden="true">&middot;</span>
          <TextLink href={LINKS.models}>Models</TextLink>
        </div>
      </nav>

      <figure className="window">
        <div className="window-bar" aria-hidden="true">
          <span className="traffic red" />
          <span className="traffic amber" />
          <span className="traffic green" />
          <span className="window-title">Dynamics Needed — Reaper</span>
        </div>
        <img
          className="preview"
          src={preview}
          alt="A drum MIDI take in Reaper: flat note velocities are reshaped into an expressive, human velocity curve."
          loading="eager"
          width="1884"
          height="1454"
        />
      </figure>

      <footer className="footer">
        <span>Dynamics Needed</span>
        <span className="sep" aria-hidden="true">&middot;</span>
        <span>drum-velocity models for your DAW</span>
      </footer>
    </main>
  )
}

export default function App() {
  return <Hero />
}

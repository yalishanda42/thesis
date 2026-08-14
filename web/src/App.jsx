import { useEffect, useState } from 'react'
import preview from './assets/preview.gif'

const LINKS = {
  demo: 'https://huggingface.co/spaces/yalishanda/dynamics-needed',
  plugin: 'https://github.com/yalishanda42/dynamics-needed#install-in-reaper',
  github: 'https://github.com/yalishanda42/dynamics-needed',
  models: 'https://huggingface.co/yalishanda',
}

// Language switcher order + endonyms (each label shown in its own language).
const LANGS = [
  { code: 'en', label: 'English' },
  { code: 'zh-CN', label: '简体中文' },
  { code: 'zh-TW', label: '繁體中文' },
  { code: 'ja', label: '日本語' },
  { code: 'ko', label: '한국어' },
  { code: 'it', label: 'Italiano' },
  { code: 'fr', label: 'Français' },
  { code: 'es', label: 'Español' },
  { code: 'de', label: 'Deutsch' },
  { code: 'ru', label: 'Русский' },
  { code: 'bg', label: 'Български' },
]

// Copy per language. `eyebrow` is the "midi drums · dynamics" tagline (uppercased
// by CSS); `title` is an array of lines (each line is one visual row of the big
// display title). `github` stays "GitHub" everywhere (proper noun).
const T = {
  en: {
    eyebrow: 'midi drums · dynamics',
    title: ['Dynamics', 'Needed.'],
    subtitle: 'Humanize drum dynamics.',
    demo: 'Try the live demo',
    plugin: 'Reaper plugin',
    models: 'Models',
  },
  'zh-CN': {
    eyebrow: 'MIDI 鼓 · 力度',
    title: ['需要力度。'],
    subtitle: '为鼓的力度注入人性。',
    demo: '试用在线演示',
    plugin: 'Reaper 插件',
    models: '模型',
  },
  'zh-TW': {
    eyebrow: 'MIDI 鼓 · 力度',
    title: ['需要力度。'],
    subtitle: '為鼓的力度注入人性。',
    demo: '試用線上示範',
    plugin: 'Reaper 外掛',
    models: '模型',
  },
  ja: {
    eyebrow: 'MIDIドラム · ダイナミクス',
    title: ['ダイナミクス', 'が必要。'],
    subtitle: 'ドラムに人間らしいダイナミクスを。',
    demo: 'デモを試す',
    plugin: 'Reaper プラグイン',
    models: 'モデル',
  },
  ko: {
    eyebrow: 'MIDI 드럼 · 다이내믹스',
    title: ['다이내믹스가', '필요해.'],
    subtitle: '드럼 다이내믹스를 인간답게.',
    demo: '라이브 데모 체험',
    plugin: 'Reaper 플러그인',
    models: '모델',
  },
  it: {
    eyebrow: 'batteria midi · dinamica',
    title: ['Serve', 'Dinamica.'],
    subtitle: 'Umanizza la dinamica della batteria.',
    demo: 'Prova la demo live',
    plugin: 'Plugin Reaper',
    models: 'Modelli',
  },
  fr: {
    eyebrow: 'batterie midi · dynamique',
    title: ['Dynamique', 'Requise.'],
    subtitle: 'Humanisez la dynamique de la batterie.',
    demo: 'Essayer la démo',
    plugin: 'Plugin Reaper',
    models: 'Modèles',
  },
  es: {
    eyebrow: 'batería midi · dinámica',
    title: ['Dinámica', 'Necesaria.'],
    subtitle: 'Humaniza la dinámica de la batería.',
    demo: 'Prueba la demo',
    plugin: 'Plugin de Reaper',
    models: 'Modelos',
  },
  de: {
    eyebrow: 'midi-schlagzeug · dynamik',
    title: ['Dynamik', 'Gefragt.'],
    subtitle: 'Mach die Schlagzeugdynamik menschlich.',
    demo: 'Live-Demo testen',
    plugin: 'Reaper-Plugin',
    models: 'Modelle',
  },
  ru: {
    eyebrow: 'midi-барабаны · динамика',
    title: ['Нужна', 'Динамика.'],
    subtitle: 'Очеловечьте динамику барабанов.',
    demo: 'Попробовать демо',
    plugin: 'Плагин Reaper',
    models: 'Модели',
  },
  bg: {
    eyebrow: 'midi барабани · динамика',
    title: ['Динамика', 'Трябва.'],
    subtitle: 'Подобри динамиката на барабаните.',
    demo: 'Пробвай демото',
    plugin: 'Reaper плъгин',
    models: 'Модели',
  },
}

// Pick a sensible default: an explicit ?lang= override wins, then the browser
// preference, then English.
function detectLang() {
  if (typeof window !== 'undefined') {
    const q = new URLSearchParams(window.location.search).get('lang')
    if (q && T[q]) return q
  }
  if (typeof navigator === 'undefined') return 'en'
  for (const raw of navigator.languages || [navigator.language || 'en']) {
    const tag = raw.toLowerCase()
    if (tag.startsWith('zh')) {
      return /tw|hk|hant|mo/.test(tag) ? 'zh-TW' : 'zh-CN'
    }
    const base = tag.split('-')[0]
    const hit = LANGS.find((l) => l.code === base)
    if (hit) return hit.code
  }
  return 'en'
}

function TextLink({ href, children }) {
  return (
    <a className="text-link" href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  )
}

function Hero() {
  const [lang, setLang] = useState(detectLang)
  const t = T[lang] ?? T.en

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  return (
    <main className="hero">
      <div className="glow" aria-hidden="true" />

      <div className="lang">
        <select
          className="lang-select"
          aria-label="Language"
          value={lang}
          onChange={(e) => setLang(e.target.value)}
        >
          {LANGS.map((l) => (
            <option key={l.code} value={l.code}>
              {l.label}
            </option>
          ))}
        </select>
      </div>

      <p className="eyebrow">{t.eyebrow}</p>

      <h1 className="title">
        {t.title.map((line, i) => (
          <span key={i}>
            {line}
            {i < t.title.length - 1 && <br />}
          </span>
        ))}
      </h1>

      <p className="subtitle">{t.subtitle}</p>

      <nav className="cta-row" aria-label="Primary">
        <a className="cta-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
          {t.demo}
        </a>
        <div className="cta-links">
          <TextLink href={LINKS.plugin}>{t.plugin}</TextLink>
          <span className="dot" aria-hidden="true">&middot;</span>
          <TextLink href={LINKS.github}>GitHub</TextLink>
          <span className="dot" aria-hidden="true">&middot;</span>
          <TextLink href={LINKS.models}>{t.models}</TextLink>
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
        <span>&copy; 2026 yalishanda (Alexander Ignatov)</span>
      </footer>
    </main>
  )
}

export default function App() {
  return <Hero />
}

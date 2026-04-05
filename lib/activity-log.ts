export type InferenceEvent = {
  id: string
  timestamp: string // ISO
  ffteFeedSolidsSP: number
  ffteProductionSolidsSP: number
  ffteSteamPressureSP: number
  tfeOutFlowSP: number
  tfeProductionSolidsSP: number
  tfeVacuumPressureSP: number
  tfeSteamPressureSP: number
  prediction: 'GOOD' | 'LOW_BAD' | 'HIGH_BAD'
  pGood: number        // 0.0–1.0
  pDowntime: number    // 0.0–1.0
  downtimeRisk: number // 0–100
  recPGood: number
  recPDowntimeRisk: number
  recommended: {
    ffteFeedSolidsSP: number
    ffteProductionSolidsSP: number
    ffteSteamPressureSP: number
    tfeOutFlowSP: number
    tfeProductionSolidsSP: number
    tfeVacuumPressureSP: number
    tfeSteamPressureSP: number
  }
}

const events: InferenceEvent[] = []
const MAX_EVENTS = 1000

export function logInference(event: Omit<InferenceEvent, 'id' | 'timestamp'>) {
  const full: InferenceEvent = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
    ...event,
  }
  events.push(full)
  if (events.length > MAX_EVENTS) {
    events.splice(0, events.length - MAX_EVENTS)
  }
  return full
}

export function getRecentEvents(limit = 50): InferenceEvent[] {
  return events.slice(-limit).reverse()
}

export function getEventsSince(msAgo: number): InferenceEvent[] {
  const cutoff = Date.now() - msAgo
  return events.filter((e) => new Date(e.timestamp).getTime() >= cutoff)
}


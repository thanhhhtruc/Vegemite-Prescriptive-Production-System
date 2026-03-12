'use client'

import { useEffect, useState } from 'react'
import { MachineInput, type AllSPInputs, type RecommendSpResult } from '@/components/machine-input'
import { StatusCard } from '@/components/status-card'
import { RecommendedSP } from '@/components/recommended-sp'
import { SensorStabilityChart } from '@/components/sensor-stability-chart'
import { SPDeviationChart } from '@/components/sp-deviation-chart'

function formatLastUpdated(date: Date) {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins} min ago`
  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return `${diffHours} hr ago`
  return date.toLocaleDateString()
}

type SPKey = keyof Omit<AllSPInputs, 'part'>

const SP_KEYS: SPKey[] = [
  'ffteFeedSolidsSP', 'ffteProductionSolidsSP', 'ffteSteamPressureSP',
  'tfeOutFlowSP', 'tfeProductionSolidsSP', 'tfeVacuumPressureSP', 'tfeSteamPressureSP',
]

type InferenceResult = RecommendSpResult & {
  prediction: 'GOOD' | 'LOW_BAD' | 'HIGH_BAD'
  downtimeRisk: number
  recommendedPGood: number
  recommendedPDowntime: number
}

type LiveDataRow = {
  cursor: number
  total: number
  timestamp: string
  part: string
  sp: Omit<AllSPInputs, 'part'>
  sensors: Record<string, number>
  quality: string
  batch: string
}

async function fetchLiveRow(): Promise<LiveDataRow> {
  const res = await fetch('/api/live-data')
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<LiveDataRow>
}

async function runInference(sp: AllSPInputs, sensors?: Record<string, number>): Promise<InferenceResult> {
  const body = { ...sp, sensors }
  const res = await fetch('/api/recommend-sp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<InferenceResult>
}

export default function DashboardPage() {
  const [inputs, setInputs] = useState<AllSPInputs | null>(null)
  const [lastUpdated, setLastUpdated] = useState(() => new Date())
  const [isManualMode, setIsManualMode] = useState(false)
  const [initialized, setInitialized] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [liveRow, setLiveRow] = useState<LiveDataRow | null>(null)

  async function infer(sp: AllSPInputs, sensorsOverride?: Record<string, number>) {
    try {
      // Use sensors from liveRow if not explicitly provided (e.g. in tick)
      const sensors = sensorsOverride ?? liveRow?.sensors
      const r = await runInference(sp, sensors)
      setResult(r)
      setInitialized(true)
      setLastUpdated(new Date())
    } catch (e) {
      console.error('Inference failed:', e)
    }
  }

  // Manual apply: user edits SPs and applies
  async function handleSetInputs(newInputs: AllSPInputs) {
    setInputs(newInputs)
    setIsManualMode(false)
    // Pass last known sensors
    await infer(newInputs)
  }

  // Apply recommended SPs
  async function handleApplyRecommended() {
    if (!result || !inputs) return
    const sp: AllSPInputs = { ...result.recommendedSP, part: inputs.part }
    setInputs(sp)
    // Pass last known sensors
    await infer(sp)
  }

  // Live loop: fetch next CSV row every 5s, run inference on real data
  useEffect(() => {
    const tick = async () => {
      if (isManualMode) return
      try {
        const row = await fetchLiveRow()
        setLiveRow(row)
        const sp: AllSPInputs = { ...row.sp, part: row.part }
        setInputs(sp)
        await infer(sp, row.sensors)
      } catch (e) {
        console.error('Live data fetch failed:', e)
      }
    }

    void tick()
    const interval = setInterval(tick, 5000)
    return () => clearInterval(interval)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isManualMode])

  const currentInputs: AllSPInputs = inputs ?? {
    ffteFeedSolidsSP: 50, ffteProductionSolidsSP: 41.5, ffteSteamPressureSP: 113,
    tfeOutFlowSP: 2400, tfeProductionSolidsSP: 65, tfeVacuumPressureSP: -68,
    tfeSteamPressureSP: 119, part: 'Yeast - BRD',
  }

  const recommendations = Object.fromEntries(
    SP_KEYS.map((k) => [k, {
      old: currentInputs[k] as number,
      new: result?.recommendedSP[k] ?? currentInputs[k] as number,
    }])
  ) as Record<SPKey, { old: number; new: number }>

  return (
    <main className="p-4 lg:p-6">
      <div className="mx-auto w-full max-w-[90rem] space-y-6">

        {/* Header */}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-foreground">Production Dashboard</h1>
            <p className="mt-0.5 text-sm text-muted-foreground">
              Quality prediction, downtime risk &amp; setpoint recommendations
            </p>
          </div>
          <div className="flex flex-col items-end gap-0.5">
            <p className="text-xs tabular-nums text-muted-foreground">
              Last updated: {formatLastUpdated(lastUpdated)}
            </p>
            {liveRow && (
              <p className="text-[10px] tabular-nums text-muted-foreground">
                Row {liveRow.cursor}/{liveRow.total} · {liveRow.timestamp}
                {liveRow.batch ? ` · Batch ${liveRow.batch}` : ''}
              </p>
            )}
          </div>
        </div>

        {/* Main grid */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 lg:gap-5">
          {/* Machine Input — spans 2 cols */}
          <div className="sm:col-span-2">
            <MachineInput
              inputs={currentInputs}
              setInputs={handleSetInputs}
              isManualMode={isManualMode}
              onToggleManual={() => setIsManualMode((v) => !v)}
              onApplyRecommended={handleApplyRecommended}
              hasRecommendation={!!result}
              currentState={initialized ? result?.prediction : undefined}
              loading={!initialized}
            />
          </div>

          {/* Status Card */}
          <div>
            <StatusCard
              prediction={result?.prediction ?? 'GOOD'}
              confidence={result?.pGood ?? 0}
              downtimeRisk={result ? Math.round(result.downtimeRisk) : 0}
              loading={!initialized}
            />
          </div>

          {/* Recommended SP */}
          <div>
            <RecommendedSP
              recommendations={recommendations}
              prescriptive={
                result
                  ? {
                      current: Object.fromEntries(SP_KEYS.map((k) => [k, currentInputs[k]])) as Omit<AllSPInputs, 'part'>,
                      recommended: result.recommendedSP,
                      pGood: result.recommendedPGood,
                      pDowntime: result.recommendedPDowntime,
                    }
                  : undefined
              }
            />
          </div>
        </div>

        {/* Charts — 2 col */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:gap-5">
          <div className="flex min-h-[360px] flex-col">
            <SensorStabilityChart
              pGood={result?.pGood}
              pDowntime={result?.pDowntime}
              dataTimestamp={liveRow?.timestamp}
            />
          </div>
          <div className="flex min-h-[360px] flex-col">
            <SPDeviationChart
              recommendations={recommendations}
              loading={!initialized}
            />
          </div>
        </div>

        <footer className="border-t pb-2 pt-6 text-center text-xs text-muted-foreground">
          Prescriptive Production System · Theme 3 · Replaying {liveRow?.total ?? '~30k'} rows from production dataset
        </footer>
      </div>
    </main>
  )
}

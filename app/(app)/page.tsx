'use client'

import { useEffect, useState, useRef } from 'react'
import { MachineInput, type AllSPInputs, type RecommendSpResult } from '@/components/machine-input'
import { StatusCard } from '@/components/status-card'
import { RecommendedSP } from '@/components/recommended-sp'
import { SensorStabilityChart } from '@/components/sensor-stability-chart'
import { SPDeviationChart } from '@/components/sp-deviation-chart'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'

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

const RECIPE_BASELINES: Record<string, Omit<AllSPInputs, 'part' | 'extractTankLevel'>> = {
  "Yeast - FMX": {
    "ffteFeedSolidsSP": 50.0,
    "ffteProductionSolidsSP": 42.9,
    "ffteSteamPressureSP": 115.0,
    "tfeOutFlowSP": 2897.65,
    "tfeProductionSolidsSP": 71.0,
    "tfeVacuumPressureSP": -62.66,
    "tfeSteamPressureSP": 120.0
  },
  "Yeast - BRD": {
    "ffteFeedSolidsSP": 50.0,
    "ffteProductionSolidsSP": 41.66,
    "ffteSteamPressureSP": 119.0,
    "tfeOutFlowSP": 2174.46,
    "tfeProductionSolidsSP": 63.0,
    "tfeVacuumPressureSP": -74.62,
    "tfeSteamPressureSP": 120.0
  },
  "Yeast - BRN": {
    "ffteFeedSolidsSP": 50.0,
    "ffteProductionSolidsSP": 41.5,
    "ffteSteamPressureSP": 120.0,
    "tfeOutFlowSP": 2081.93,
    "tfeProductionSolidsSP": 65.0,
    "tfeVacuumPressureSP": -71.62,
    "tfeSteamPressureSP": 120.0
  }
}

type InferenceResult = RecommendSpResult & {
  prediction: string
  downtimeRisk: number
  rootCause?: string[]
  isoAnomaly?: boolean
  recommendedPGood: number
  recommendedPDowntime: number
}

type LiveDataRow = {
  cursor: number
  total: number
  batchCursor: number
  batchTotal: number
  timestamp: string
  part: string
  sp: Omit<AllSPInputs, 'part'>
  sensors: Record<string, number>
  criticalSensors: string[]
  quality: string
  batch: string
}

async function fetchLiveRow(step: number = 80): Promise<LiveDataRow> {
  const res = await fetch(`/api/live-data?step=${step}`)
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
  const inputsRef = useRef<AllSPInputs | null>(null)
  const [lastUpdated, setLastUpdated] = useState(() => new Date())
  const [isManualMode, setIsManualMode] = useState(false)
  const isManualModeRef = useRef(false)
  const [initialized, setInitialized] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const resultRef = useRef<InferenceResult | null>(null)
  const [liveRow, setLiveRow] = useState<LiveDataRow | null>(null)
  const [processState, setProcessState] = useState<'RUNNING' | 'CHANGEOVER' | 'READY' | 'HALTED'>('RUNNING')
  const [cipProgress, setCipProgress] = useState(0)

  const currentBatchRef = useRef<string | null>(null)
  const processStateRef = useRef<'RUNNING' | 'CHANGEOVER' | 'READY' | 'HALTED'>('RUNNING')

  async function infer(sp: AllSPInputs, sensorsOverride?: Record<string, number>) {
    try {
      // Use sensors from liveRow if not explicitly provided (e.g. in tick)
      const sensors = sensorsOverride ?? liveRow?.sensors
      const r = await runInference(sp, sensors)
      setResult(r)
      resultRef.current = r
      setInitialized(true)
      setLastUpdated(new Date())
      return r
    } catch (e) {
      console.error('Inference failed:', e)
      return null
    }
  }

  // Manual apply: user edits SPs and applies
  async function handleSetInputs(newInputs: AllSPInputs) {
    setInputs(newInputs)
    inputsRef.current = newInputs
    setIsManualMode(false)
    isManualModeRef.current = false
    // Pass last known sensors + override extractTankLevel
    const sensors = { ...liveRow?.sensors }
    if (newInputs.extractTankLevel !== undefined) {
      sensors['Extract tank Level'] = newInputs.extractTankLevel
    }
    const r = await infer(newInputs, sensors)
    
    // Auto-resume if user manually overrides during HALTED state and it's resolved
    if (processStateRef.current === 'HALTED' && r && r.downtimeRisk < 30) {
      processStateRef.current = 'RUNNING'
      setProcessState('RUNNING')
    }
  }

  // Apply recommended SPs
  async function handleApplyRecommended() {
    if (!result || !inputs) return
    const sp: AllSPInputs = { ...result.recommendedSP, part: inputs.part, extractTankLevel: inputs.extractTankLevel }
    setInputs(sp)
    inputsRef.current = sp
    // Pass last known sensors + override extractTankLevel
    const sensors = { ...liveRow?.sensors }
    if (inputs.extractTankLevel !== undefined) {
      sensors['Extract tank Level'] = inputs.extractTankLevel
    }
    const r = await infer(sp, sensors)

    // Auto-resume if recommended settings resolve the halting condition
    if (processStateRef.current === 'HALTED' && r && r.downtimeRisk < 30) {
      processStateRef.current = 'RUNNING'
      setProcessState('RUNNING')
    }
  }

  // Live loop: fetch next CSV row every 6s, run inference on real data
  useEffect(() => {
    let isSubscribed = true
    let timeoutId: NodeJS.Timeout

    const tick = async () => {
      if (!isSubscribed) return
      
      // Auto-pause when entering Manual Mode
      if (isManualModeRef.current || processStateRef.current !== 'RUNNING') {
        timeoutId = setTimeout(tick, 2000)
        return
      }

      try {
        const isRisky = resultRef.current && resultRef.current.downtimeRisk > 15
        const stepSize = isRisky ? 10 : 80
        const row = await fetchLiveRow(stepSize)
        
        // Detect if the batch has changed
        if (currentBatchRef.current && currentBatchRef.current !== row.batch) {
          console.log(`[Batch Change] ${currentBatchRef.current} -> ${row.batch}`)
          
          processStateRef.current = 'CHANGEOVER'
          setProcessState('CHANGEOVER')
          setCipProgress(0)

          // Clear buffer in backend
          await fetch('/api/reset-buffer', { method: 'POST' })

          // Simulate CIP progress over ~3 seconds
          let p = 0
          const cipInterval = setInterval(() => {
            p += 10
            setCipProgress(p)
            if (p >= 100) {
              clearInterval(cipInterval)
              currentBatchRef.current = row.batch
              // Tự động vào luôn RUNNING thay vì bắt đợi READY
              processStateRef.current = 'RUNNING'
              setProcessState('RUNNING')
              timeoutId = setTimeout(tick, 1000)
            }
          }, 300)
          return
        }

        const isNewBatch = currentBatchRef.current !== row.batch
        currentBatchRef.current = row.batch
        setLiveRow(row)

        let sp: AllSPInputs
        if (isNewBatch || !inputsRef.current) {
          // Initialize with Golden Batch recipe baseline
          const baseline = RECIPE_BASELINES[row.part] || RECIPE_BASELINES['Yeast - FMX']
          sp = { ...baseline, part: row.part, extractTankLevel: row.sensors['Extract tank Level'] ?? 65 }
        } else {
          // Carry over current operational state
          sp = { ...inputsRef.current, part: row.part, extractTankLevel: row.sensors['Extract tank Level'] ?? inputsRef.current.extractTankLevel ?? 65 }
        }
        
        setInputs(sp)
        inputsRef.current = sp
        const inferredResult = await infer(sp, row.sensors)

        if (inferredResult && inferredResult.downtimeRisk >= 30) {
          processStateRef.current = 'HALTED'
          setProcessState('HALTED')
        }
      } catch (e) {
        console.error('Live data fetch failed:', e)
      }

      timeoutId = setTimeout(tick, 3500)
    }

    void tick()
    return () => {
      isSubscribed = false
      clearTimeout(timeoutId)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const currentInputs: AllSPInputs = inputs ?? {
    ffteFeedSolidsSP: 50, ffteProductionSolidsSP: 41.5, ffteSteamPressureSP: 113,
    tfeOutFlowSP: 2400, tfeProductionSolidsSP: 65, tfeVacuumPressureSP: -68,
    tfeSteamPressureSP: 119, part: 'Yeast - BRD', extractTankLevel: liveRow?.sensors['Extract tank Level'] ?? 65,
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
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between w-full">
          <div className="flex-1 w-full sm:max-w-2xl">
            <h1 className="text-xl font-semibold tracking-tight text-foreground">Production Dashboard</h1>
            <p className="mt-0.5 text-sm text-muted-foreground mb-4">
              Quality prediction, downtime risk &amp; setpoint recommendations
            </p>
            
            {/* New specific batch and progress aligned left for better visibility */}
            {processState === 'RUNNING' && (
              <div className="flex flex-col gap-1.5 w-full">
                <div className="flex items-center gap-2">
                  <span className="flex size-2 rounded-full bg-emerald-500 animate-pulse"></span>
                  <span className="text-sm font-bold uppercase tracking-widest text-emerald-600">
                    Running | Batch: {liveRow?.batch ?? 'WAITING...'}
                  </span>
                </div>
                {liveRow && (
                  <div className="flex items-center gap-3 w-full">
                    <Progress value={(liveRow.batchCursor / liveRow.batchTotal) * 100} className="h-2 flex-1 bg-muted" />
                    <span className="text-xs font-medium tabular-nums text-muted-foreground whitespace-nowrap">
                      {(liveRow.batchCursor / liveRow.batchTotal * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>
            )}
            
            {processState === 'CHANGEOVER' && (
              <div className="flex items-center gap-2">
                <span className="flex size-2 rounded-full bg-amber-400 animate-bounce"></span>
                <span className="text-sm font-bold uppercase tracking-widest text-amber-600">
                   CIP | Preparing New Batch
                </span>
              </div>
            )}

            {processState === 'READY' && (
              <div className="flex items-center gap-3">
                <span className="text-sm font-bold uppercase tracking-widest text-blue-600">
                   Batch Ready
                </span>
                <Button 
                  size="sm" 
                  onClick={() => {
                    processStateRef.current = 'RUNNING'
                    setProcessState('RUNNING')
                  }}
                >
                  Start Batch
                </Button>
              </div>
            )}

            {processState === 'HALTED' && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="flex size-2 rounded-full bg-destructive animate-pulse"></span>
                  <span className="text-sm font-bold uppercase tracking-widest text-destructive">
                     Production Halted
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Button 
                    size="sm" 
                    variant="destructive"
                    onClick={() => {
                      processStateRef.current = 'RUNNING'
                      setProcessState('RUNNING')
                    }}
                  >
                    Force Resume (Same Batch)
                  </Button>
                  <Button 
                    size="sm" 
                    variant="outline"
                    className="border-destructive/30 hover:bg-destructive/10 text-destructive"
                    onClick={async () => {
                      // Bỏ qua mẻ lỗi, tua nhanh API data đến mẻ kế tiếp
                      await fetch('/api/live-data', {
                        method: 'POST',
                        body: JSON.stringify({ action: 'skip-to-next-batch' }),
                        headers: { 'Content-Type': 'application/json' }
                      })
                      processStateRef.current = 'RUNNING'
                      setProcessState('RUNNING')
                    }}
                  >
                    Abandon & Skip to Next Batch
                  </Button>
                </div>
              </div>
            )}
          </div>
          
          <div className="flex flex-col items-end gap-1 mt-2 sm:mt-0">
            <p className="text-[10px] tabular-nums text-muted-foreground">
              Last updated: {formatLastUpdated(lastUpdated)}
            </p>
            {liveRow && (
              <p className="text-[10px] tabular-nums text-muted-foreground">
                Row {liveRow.batchCursor}/{liveRow.batchTotal} · {liveRow.timestamp}
              </p>
            )}
          </div>
        </div>

        {processState === 'CHANGEOVER' && (
          <div className="w-full space-y-1.5 animate-in fade-in slide-in-from-top-1">
            <div className="flex justify-between text-xs text-amber-600 font-semibold uppercase tracking-wider">
              <span>Flushing Sensor Buffer &amp; Lines...</span>
              <span>{Math.min(cipProgress, 100)}%</span>
            </div>
            <Progress value={cipProgress} className="h-2 w-full" />
          </div>
        )}

        {/* Main grid */}
        <div className={`grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 lg:gap-5 transition-all duration-500 relative ${processState === 'CHANGEOVER' ? 'opacity-50 pointer-events-none grayscale saturate-50' : ''}`}>
          {processState === 'CHANGEOVER' && (
            <div className="absolute inset-0 z-50 flex items-center justify-center">
              <div className="bg-background/80 px-4 py-2 text-sm font-semibold rounded shadow-sm border border-border/50 text-amber-600 animate-pulse">
                System Initializing... AI Warming Up...
              </div>
            </div>
          )}
          {/* Machine Input — spans 2 cols */}
          <div className="sm:col-span-2">
            <MachineInput
              inputs={currentInputs}
              setInputs={handleSetInputs}
              isManualMode={isManualMode}
              onToggleManual={() => {
                setIsManualMode((v) => !v)
                isManualModeRef.current = !isManualModeRef.current
              }}
              onApplyRecommended={handleApplyRecommended}
              hasRecommendation={!!result}
              currentState={initialized ? result?.prediction : undefined}
              loading={!initialized || processState === 'CHANGEOVER'}
            />
          </div>

          {/* Status Card */}
          <div>
            <StatusCard
              prediction={result?.prediction ?? 'GOOD'}
              confidence={result?.pGood ?? 0}
              downtimeRisk={result ? Math.round(result.downtimeRisk) : 0}
              rootCause={result?.rootCause}
              isoAnomaly={result?.isoAnomaly}
              loading={!initialized || processState === 'CHANGEOVER'}
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
        <div className={`grid grid-cols-1 gap-4 lg:grid-cols-2 lg:gap-5 transition-all duration-500 relative ${processState === 'CHANGEOVER' ? 'opacity-50 pointer-events-none grayscale saturate-50' : ''}`}>
          {processState === 'CHANGEOVER' && (
            <div className="absolute inset-0 z-50 flex items-center justify-center">
               <div className="bg-background/80 px-4 py-2 text-sm font-semibold rounded shadow-sm border border-border/50 text-amber-600 animate-pulse">
                Building Baseline for Next Batch...
               </div>
            </div>
          )}
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
              loading={!initialized || processState === 'CHANGEOVER'}
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

'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Label } from '@/components/ui/label'
import { Loader2 } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

export type AllSPInputs = {
  ffteFeedSolidsSP: number
  ffteProductionSolidsSP: number
  ffteSteamPressureSP: number
  tfeOutFlowSP: number
  tfeProductionSolidsSP: number
  tfeVacuumPressureSP: number
  tfeSteamPressureSP: number
  part: string
}

export type RecommendSpResult = {
  recommendedSP: Omit<AllSPInputs, 'part'>
  pGood: number
  pDowntime: number
}

interface MachineInputProps {
  inputs: AllSPInputs
  setInputs?: (inputs: AllSPInputs) => void
  isManualMode?: boolean
  onToggleManual?: () => void
  onApplyRecommended?: () => void
  hasRecommendation?: boolean
  currentState?: 'GOOD' | 'LOW_BAD' | 'HIGH_BAD'
  loading?: boolean
  hidePart?: boolean
}

const PARTS = ['Yeast - BRD', 'Yeast - BRN', 'Yeast - FMX']

const FIELDS: { key: keyof Omit<AllSPInputs, 'part'>; label: string; unit: string }[] = [
  { key: 'ffteFeedSolidsSP',       label: 'FFTE Feed Solids SP',       unit: '%' },
  { key: 'ffteProductionSolidsSP', label: 'FFTE Production Solids SP', unit: '%' },
  { key: 'ffteSteamPressureSP',    label: 'FFTE Steam Pressure SP',    unit: 'kPa' },
  { key: 'tfeOutFlowSP',           label: 'TFE Out Flow SP',           unit: 'L/h' },
  { key: 'tfeProductionSolidsSP',  label: 'TFE Production Solids SP',  unit: '%' },
  { key: 'tfeVacuumPressureSP',    label: 'TFE Vacuum Pressure SP',    unit: 'kPa' },
  { key: 'tfeSteamPressureSP',     label: 'TFE Steam Pressure SP',     unit: 'kPa' },
]

const STATE_CONFIG = {
  GOOD:     { label: 'Good',     variant: 'default'     as const, dot: 'bg-emerald-500' },
  LOW_BAD:  { label: 'Low Bad',  variant: 'secondary'   as const, dot: 'bg-amber-400' },
  HIGH_BAD: { label: 'High Bad', variant: 'destructive' as const, dot: 'bg-red-500' },
}

export function MachineInput({
  inputs,
  setInputs,
  isManualMode = false,
  onToggleManual,
  onApplyRecommended,
  hasRecommendation = false,
  currentState,
  loading = false,
  hidePart = false,
}: MachineInputProps) {
  const [localInputs, setLocalInputs] = useState<any>(inputs)
  const [isApplying, setIsApplying] = useState(false)

  useEffect(() => {
    setLocalInputs(inputs)
    setIsApplying(false)
  }, [inputs])

  const handleInputChange = (key: keyof Omit<AllSPInputs, 'part'>, value: string) => {
    setLocalInputs((prev: any) => ({ ...prev, [key]: value }))
  }

  const handleBlur = (key: keyof Omit<AllSPInputs, 'part'>) => {
    setLocalInputs((prev: any) => ({
      ...prev,
      [key]: prev[key] === '' || isNaN(Number(prev[key])) ? 0 : Number(prev[key])
    }))
  }

  const handlePartChange = (value: string) => {
    setLocalInputs((prev: any) => ({ ...prev, part: value }))
  }

  const handleApply = () => {
    setIsApplying(true)
    const parsedInputs = { ...localInputs }
    FIELDS.forEach(({ key }) => {
      parsedInputs[key] = Number(parsedInputs[key]) || 0
    })
    
    // Simulate a small delay to show the loading animation
    setTimeout(() => {
      if (setInputs) setInputs(parsedInputs)
      if (onToggleManual) onToggleManual()
      setIsApplying(false)
    }, 400)
  }

  const handleCancel = () => {
    setLocalInputs(inputs)
    if (onToggleManual) onToggleManual()
  }

  const stateConfig = currentState ? STATE_CONFIG[currentState] : null

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-sm font-semibold tracking-tight">Machine Setpoints</CardTitle>
          {loading ? (
            <Badge variant="outline" className="animate-pulse text-xs">Initializing…</Badge>
          ) : stateConfig ? (
            <Badge variant={stateConfig.variant} className="gap-1.5 text-xs font-semibold">
              <span className={`inline-block size-1.5 rounded-full ${stateConfig.dot}`} />
              {stateConfig.label}
            </Badge>
          ) : null}
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col justify-between gap-4 p-4 pt-2">
        {/* Yeast type selector — hidden when part shown in section header */}
        {!hidePart && (
          <div className="flex flex-col gap-2">
            <Label className="text-xs font-semibold text-muted-foreground">Yeast Type (Batch)</Label>
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center rounded-md bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                {inputs.part}
              </span>
              <span className="text-[10px] text-muted-foreground italic">
                Auto-detected from line
              </span>
            </div>
          </div>
        )}

        <Separator />


        {/* SP fields — 2-col grid */}
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
          {FIELDS.map(({ key, label, unit }) => (
            <div key={key} className="flex flex-col gap-1 rounded-lg border bg-muted/40 p-2.5">
              <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground leading-tight">
                {label}
              </span>
              <div className="flex items-baseline gap-1">
                {isManualMode ? (
                  <Input
                    type="number"
                    value={localInputs[key]}
                    onChange={(e) => handleInputChange(key, e.target.value)}
                    onBlur={() => handleBlur(key)}
                    step="0.1"
                    className="h-7 w-full text-base font-bold tabular-nums tracking-tight bg-background px-1.5"
                  />
                ) : (
                  <span className="text-lg font-bold tabular-nums tracking-tight">
                    {(inputs[key] as number).toFixed(1)}
                  </span>
                )}
                <span className="text-[10px] font-medium text-muted-foreground">{unit}</span>
              </div>
            </div>
          ))}
        </div>

        <Separator />

        {/* Action buttons row */}
        <div className="flex items-center gap-2">
          {isManualMode ? (
            <>
              <Button size="sm" onClick={handleApply} disabled={isApplying} className="h-8 text-xs min-w-[100px] transition-all">
                {isApplying ? (
                  <>
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> Fetching...
                  </>
                ) : (
                  <>Apply &amp; Run</>
                )}
              </Button>
              <Button size="sm" variant="outline" onClick={handleCancel} disabled={isApplying} className="h-8 text-xs">
                Cancel
              </Button>
            </>
          ) : (
            <Button size="sm" variant="outline" onClick={onToggleManual} className="h-8 text-xs">
              Edit Setpoints
            </Button>
          )}
          {hasRecommendation && onApplyRecommended && (
            <Button
              size="sm"
              onClick={onApplyRecommended}
              style={{ backgroundColor: '#FFEA17', color: '#000000' }}
              className="h-8 text-xs hover:opacity-90 border-0"
            >
              Use Recommended
            </Button>
          )}
          <p className="ml-auto text-[10px] text-muted-foreground">
            {isManualMode ? 'Editing — apply to run inference' : 'Live setpoints from sensors'}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

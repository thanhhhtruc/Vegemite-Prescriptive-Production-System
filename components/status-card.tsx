import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'

interface StatusCardProps {
  prediction: string
  confidence: number
  downtimeRisk: number
  rootCause?: string[]
  isoAnomaly?: boolean
  loading?: boolean
}

const PREDICTION_CONFIG: Record<string, { label: string, variant: 'default' | 'secondary' | 'destructive' | 'outline', numClass: string, barClass: string, desc: string }> = {
  GOOD: {
    label: 'Good',
    variant: 'default',
    // Use theme foreground (dark brown) for "good" — neutral, on-brand
    numClass: 'text-foreground',
    barClass: 'bg-foreground',
    desc: 'Product meets quality spec. Solids concentration and evaporation rate are within target range.',
  },
  LOW_BAD: {
    label: 'Low Bad',
    variant: 'secondary',
    // Bright orange — distinct from red, visible in both themes
    numClass: 'text-orange-500',
    barClass: 'bg-orange-500',
    desc: 'Product solids below target. Feed rate or steam pressure may need adjustment.',
  },
  HIGH_BAD: {
    label: 'High Bad',
    variant: 'destructive',
    // Theme primary = Vegemite red
    numClass: 'text-primary',
    barClass: 'bg-primary',
    desc: 'Product solids above target. Reduce TFE out-flow or lower steam pressure SP.',
  },
  UNKNOWN: {
    label: 'Unknown',
    variant: 'outline',
    numClass: 'text-muted-foreground',
    barClass: 'bg-muted-foreground',
    desc: 'Status unknown. Prediction may be initializing or missing model info.',
  },
  ERROR: {
    label: 'Error',
    variant: 'destructive',
    numClass: 'text-destructive',
    barClass: 'bg-destructive',
    desc: 'An error occurred during prediction.',
  },
}

function getRiskConfig(risk: number, rootCause?: string[], isoAnomaly?: boolean) {
  let extraDesc = ''
  
  if (rootCause && rootCause.length > 0) {
    const activeCauses = rootCause.filter(c => c !== 'Normal' && c !== 'Unknown')
    if (activeCauses.length > 0) {
      extraDesc = ` Root Cause: ${activeCauses.map(c => c.replace(/_/g, ' ')).join(' | ')}.`
    }
  }

  if (isoAnomaly) {
    extraDesc += ' System detected anomalous operating conditions.'
  }

  if (risk >= 50) return {
    numClass: 'text-primary',
    barClass: 'bg-primary',
    badge: 'destructive' as const,
    label: 'High',
    desc: `Unplanned stoppage likely. Check evaporator train and separator status immediately.${extraDesc}`,
  }
  if (risk >= 25) return {
    numClass: 'text-orange-500',
    barClass: 'bg-orange-500',
    badge: 'secondary' as const,
    label: 'Moderate',
    desc: `Elevated risk. Monitor TFE vacuum pressure and motor current closely.${extraDesc}`,
  }
  return {
    numClass: 'text-foreground',
    barClass: 'bg-foreground',
    badge: 'outline' as const,
    label: 'Low',
    desc: 'System stable. No downtime indicators detected in current operating window.',
  }
}

export function StatusCard({ prediction, confidence, downtimeRisk, rootCause, isoAnomaly, loading }: StatusCardProps) {
  const cfg = PREDICTION_CONFIG[prediction] || PREDICTION_CONFIG.UNKNOWN
  const risk = getRiskConfig(downtimeRisk, rootCause, isoAnomaly)

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold tracking-tight">Production Status</CardTitle>
          {loading ? (
            <Badge variant="outline" className="animate-pulse text-xs">Initializing…</Badge>
          ) : (
            <Badge variant={cfg.variant} className="text-xs font-semibold">
              {cfg.label}
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col p-4 pt-2">
        {loading ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-2">
            <div className="size-10 animate-spin rounded-full border-2 border-muted border-t-foreground" />
            <p className="text-xs text-muted-foreground">Running inference…</p>
          </div>
        ) : (
          <div className="flex flex-1 flex-col justify-between gap-4">

            {/* ── Quality Confidence ── */}
            <div className="space-y-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Quality Confidence
              </p>
              <div className="flex items-end gap-1.5">
                <span className={`text-4xl font-extrabold tabular-nums leading-none ${cfg.numClass}`}>
                  {(confidence * 100).toFixed(0)}
                </span>
                <span className="mb-0.5 text-xl font-bold text-muted-foreground">%</span>
                <Badge variant={cfg.variant} className="mb-0.5 ml-auto text-[10px]">
                  {cfg.label}
                </Badge>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${cfg.barClass}`}
                  style={{ width: `${(confidence * 100).toFixed(0)}%` }}
                />
              </div>
              <p className="text-[10px] leading-snug text-muted-foreground">{cfg.desc}</p>
            </div>

            <Separator />

            {/* ── Downtime Risk ── */}
            <div className="space-y-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex justify-between">
                <span>Downtime Risk</span>
                {rootCause && rootCause.filter(c => c !== 'Normal').length > 0 && (
                  <span className="text-destructive font-bold text-right text-[10px] break-words max-w-[60%]">
                    ⚠️ {rootCause.filter(c => c !== 'Normal').map(c => c.replace(/_/g, ' ')).join(' | ')}
                  </span>
                )}
              </p>
              <div className="flex items-end gap-1.5">
                <span className={`text-4xl font-extrabold tabular-nums leading-none ${risk.numClass}`}>
                  {downtimeRisk.toFixed(0)}
                </span>
                <span className="mb-0.5 text-xl font-bold text-muted-foreground">%</span>
                <Badge variant={risk.badge} className="mb-0.5 ml-auto text-[10px]">
                  {risk.label}
                </Badge>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${risk.barClass}`}
                  style={{ width: `${Math.min(downtimeRisk, 100)}%` }}
                />
              </div>
              <p className="text-[10px] leading-snug text-muted-foreground">{risk.desc}</p>
            </div>

          </div>
        )}
      </CardContent>
    </Card>
  )
}

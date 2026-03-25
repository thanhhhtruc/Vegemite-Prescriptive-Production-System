
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, Thermometer, Droplets, Target } from "lucide-react"

interface CriticalSensorsProps {
  sensors: Record<string, number>
  criticalSensors: string[]
  loading?: boolean
}

export function CriticalSensors({ sensors, criticalSensors, loading }: CriticalSensorsProps) {
  if (loading) return (
    <Card className="w-full shadow-sm">
      <CardHeader className="p-4 pb-2">
        <CardTitle className="text-sm font-semibold">Process Indicators</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-2">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 animate-pulse">
           {[1, 2, 3, 4].map(i => <div key={i} className="h-24 bg-muted rounded-md" />)}
        </div>
      </CardContent>
    </Card>
  )

  const getIcon = (name: string) => {
    if (name.includes('temperature') || name.includes('Heat')) return <Thermometer className="w-3.5 h-3.5" />
    if (name.includes('solids') || name.includes('density')) return <Droplets className="w-3.5 h-3.5" />
    if (name.includes('SP')) return <Target className="w-3.5 h-3.5" />
    return <Activity className="w-3.5 h-3.5" />
  }

  return (
    <Card className="w-full shadow-sm">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-semibold">Process Indicators</CardTitle>
          <Badge variant="outline" className="text-[10px] font-mono h-5 bg-primary/5 uppercase tracking-wider">
            Critical
          </Badge>
        </div>
        <CardDescription className="text-[10px] sm:text-xs">
          Dynamic monitoring of part-specific critical sensors identified by AI
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-2">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {criticalSensors.slice(0, 4).map((sensor) => (
            <div 
              key={sensor} 
              className="group flex flex-col gap-1.5 p-3 rounded-lg border bg-card/50 hover:bg-accent/5 transition-all duration-200 border-primary/10 shadow-sm"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-tight text-muted-foreground">
                  {getIcon(sensor)}
                  <span className="truncate max-w-[140px]" title={sensor}>{sensor.replace(' PV', '')}</span>
                </div>
              </div>
              
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-bold font-mono tracking-tighter tabular-nums">
                  {sensors[sensor]?.toFixed(2) ?? '0.00'}
                </span>
                <span className="text-[10px] text-muted-foreground font-bold uppercase opacity-60">
                  {sensor.includes('flow') ? 'kg/h' : sensor.includes('temperature') ? '°C' : '%'}
                </span>
              </div>
              
              {/* SOTA Visual: Mini trend bar (Restored) */}
              <div className="h-1.5 w-full bg-secondary/50 rounded-full overflow-hidden mt-1">
                 <div 
                   className="h-full bg-primary/70 transition-all duration-500" 
                   style={{ width: `${Math.min(100, Math.max(10, (sensors[sensor] || 0) % 100))}%` }} 
                 />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

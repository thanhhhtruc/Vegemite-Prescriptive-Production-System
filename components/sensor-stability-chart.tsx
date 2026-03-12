'use client'

import { useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from '@/components/ui/chart'
import { CartesianGrid, Line, LineChart, XAxis, YAxis, ReferenceLine } from 'recharts'

type Point = { time: string; pGood: number; pDowntime: number }

interface SensorStabilityChartProps {
  /** Latest inference result — chart accumulates these client-side */
  pGood?: number       // 0.0–1.0
  pDowntime?: number   // 0.0–1.0
  /** Timestamp from the CSV data row (e.g. '2/07/2019 0:10') */
  dataTimestamp?: string
}

const chartConfig = {
  pGood: {
    label: 'P(Good Quality)',
    color: 'hsl(var(--chart-1))',
  },
  pDowntime: {
    label: 'Downtime Risk',
    color: 'hsl(var(--chart-2))',
  },
} satisfies ChartConfig

const MAX_POINTS = 60 // keep last 60 data points (~5 min at 5s interval)

export function SensorStabilityChart({ pGood, pDowntime, dataTimestamp }: SensorStabilityChartProps) {
  const [series, setSeries] = useState<Point[]>([])
  const tickRef = useRef(0)

  // Accumulate new points whenever inference result changes
  useEffect(() => {
    if (pGood == null || pDowntime == null) return

    console.debug(`[Chart] Adding point: pGood=${pGood}, pDT=${pDowntime}, ts=${dataTimestamp}`)

    // Use data timestamp if available, otherwise fall back to client time
    const label = dataTimestamp
      ? dataTimestamp.replace(/^\d+\/\d+\/\d+ /, '') // strip date, keep HH:MM
      : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

    setSeries((prev) => {
      // Avoid duplicate timestamps if they come too fast
      if (prev.length > 0 && prev[prev.length - 1].time === label && prev[prev.length - 1].pGood === pGood * 100) {
        return prev
      }

      const next = [
        ...prev,
        {
          time: label,
          pGood: Number((pGood * 100).toFixed(1)),
          pDowntime: Number((pDowntime * 100).toFixed(1)),
        },
      ]
      return next.slice(-MAX_POINTS)
    })
  }, [pGood, pDowntime, dataTimestamp])

  return (
    <Card className="h-full flex flex-col min-h-0">
      <CardHeader className="p-4 pb-2 space-y-1.5">
        <CardTitle className="text-sm font-semibold tracking-tight">Quality &amp; Risk Trend</CardTitle>
        <CardDescription className="text-xs">
          P(Good Quality) and Downtime Risk over time (%) — last {MAX_POINTS} inferences
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-0 flex-1 min-h-0">
        {series.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[300px] text-muted-foreground text-sm">
            Waiting for first inference…
          </div>
        ) : (
          <ChartContainer config={chartConfig} className="aspect-auto h-full min-h-[300px] w-full">
            <LineChart
              data={series}
              margin={{ left: 8, right: 12, top: 4, bottom: 0 }}
              accessibilityLayer
            >
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="time"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tick={{ fontSize: 10 }}
                // Show only every Nth tick to avoid crowding
                interval="preserveStartEnd"
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={4}
                width={36}
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 10 }}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    labelKey="time"
                    formatter={(value, _name, item) => {
                      const label = item.dataKey === 'pGood' ? 'P(Good)' : 'Downtime Risk'
                      return (
                        <>
                          <div
                            className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                            style={{ backgroundColor: item.color }}
                          />
                          <div className="flex flex-1 justify-between items-center min-w-0 gap-2">
                            <span className="text-muted-foreground">{label}</span>
                            <span className="font-mono font-medium tabular-nums text-foreground">
                              {value}%
                            </span>
                          </div>
                        </>
                      )
                    }}
                  />
                }
              />
              <ReferenceLine y={50} stroke="hsl(var(--border))" strokeDasharray="4 4" opacity={0.6} />
              <Line
                type="monotone"
                dataKey="pGood"
                stroke="hsl(var(--chart-1))"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="pDowntime"
                stroke="hsl(var(--chart-2))"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <ChartLegend content={<ChartLegendContent />} />
            </LineChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  )
}

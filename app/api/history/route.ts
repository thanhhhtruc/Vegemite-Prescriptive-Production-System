import { NextRequest, NextResponse } from 'next/server'
import { getEventsSince } from '@/lib/activity-log'

// Returns recent time-series for risk and SPs, derived from inference events.

export async function GET(request: NextRequest) {
  try {
    const windowMins = Number(request.nextUrl.searchParams.get('minutes') || '240') // last 4h
    const events = getEventsSince(windowMins * 60 * 1000)

    // Sort by timestamp ascending for proper chart display
    const sortedEvents = [...events].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )

    const riskSeries = sortedEvents.map((e) => {
      const ts = new Date(e.timestamp)
      const now = new Date()
      const diffMs = now.getTime() - ts.getTime()
      const diffSecs = Math.floor(diffMs / 1000)
      const diffMins = Math.floor(diffSecs / 60)
      
      let timeLabel: string
      if (diffSecs < 60) {
        timeLabel = `${diffSecs}s`
      } else if (diffMins < 60) {
        timeLabel = `${diffMins}m`
      } else {
        timeLabel = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
      
      return {
        time: timeLabel,
        risk: Number((e.downtimeRisk).toFixed(1)),
        pGood: Number(((e.pGood ?? 0) * 100).toFixed(1)),
      }
    })

    const sensorSeries = sortedEvents.map((e) => {
      const ts = new Date(e.timestamp)
      const now = new Date()
      const diffMs = now.getTime() - ts.getTime()
      const diffSecs = Math.floor(diffMs / 1000)
      const diffMins = Math.floor(diffSecs / 60)
      
      let timeLabel: string
      if (diffSecs < 60) {
        timeLabel = `${diffSecs}s`
      } else if (diffMins < 60) {
        timeLabel = `${diffMins}m`
      } else {
        timeLabel = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
      
      return {
        time: timeLabel,
        ffteFeedSolids: Number(e.ffteFeedSolidsSP.toFixed(1)),
        ffteProdSolids: Number(e.ffteProductionSolidsSP.toFixed(1)),
        ffteSteam: Number(e.ffteSteamPressureSP.toFixed(1)),
        tfeFlow: Number(e.tfeOutFlowSP.toFixed(1)),
        tfeProdSolids: Number(e.tfeProductionSolidsSP.toFixed(1)),
        tfeVacuum: Number(e.tfeVacuumPressureSP.toFixed(1)),
        tfeSteam: Number(e.tfeSteamPressureSP.toFixed(1)),
      }
    })

    return NextResponse.json({
      risk: riskSeries,
      sensors: sensorSeries,
    })
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Failed to load history' },
      { status: 500 },
    )
  }
}


import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import { logInference } from '@/lib/activity-log'

export type RecommendSpBody = {
  // All 7 SP inputs (correctly labeled)
  ffteFeedSolidsSP?: number
  ffteProductionSolidsSP?: number
  ffteSteamPressureSP?: number
  tfeOutFlowSP?: number
  tfeProductionSolidsSP?: number
  tfeVacuumPressureSP?: number
  tfeSteamPressureSP?: number
  // Yeast type for per-Part model routing
  part?: string
  // Real sensor PV values from live data (optional, defaults to medians if missing)
  sensors?: Record<string, number>
}

export type RecommendSpResponse = {
  recommendedSP: {
    ffteFeedSolidsSP: number
    ffteProductionSolidsSP: number
    ffteSteamPressureSP: number
    tfeOutFlowSP: number
    tfeProductionSolidsSP: number
    tfeVacuumPressureSP: number
    tfeSteamPressureSP: number
  }
  pGood: number
  pDowntime: number
  prediction: 'GOOD' | 'LOW_BAD' | 'HIGH_BAD'
  downtimeRisk: number
  recommendedPGood: number
  recommendedPDowntime: number
}

/**
 * POST /api/recommend-sp
 * Prescriptive: given all 7 SP inputs + yeast type, return recommended SPs and predictions.
 * Invokes serve_recommend_sp.py which uses per-Part XGBoost models + joint SP optimization.
 */
export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as RecommendSpBody

    const pythonPath = process.env.PYTHON_PATH || 'python'
    const scriptPath = process.cwd() + '/models/serve_recommend_sp.py'

    const payload = JSON.stringify({
      ffteFeedSolidsSP: body.ffteFeedSolidsSP,
      ffteProductionSolidsSP: body.ffteProductionSolidsSP,
      ffteSteamPressureSP: body.ffteSteamPressureSP,
      tfeOutFlowSP: body.tfeOutFlowSP,
      tfeProductionSolidsSP: body.tfeProductionSolidsSP,
      tfeVacuumPressureSP: body.tfeVacuumPressureSP,
      tfeSteamPressureSP: body.tfeSteamPressureSP,
      part: body.part ?? 'Yeast - BRD',
      sensors: body.sensors ?? {},
    })

    const result: RecommendSpResponse = await new Promise((resolve, reject) => {
      const proc = spawn(pythonPath, [scriptPath])

      let stdout = ''
      let stderr = ''

      proc.stdout.on('data', (d) => { stdout += d.toString() })
      proc.stderr.on('data', (d) => { stderr += d.toString() })
      proc.on('error', (err) => { reject(err) })
      proc.on('close', (code) => {
        if (code !== 0) {
          return reject(new Error(stderr || `Python exited with code ${code}`))
        }
        try {
          const parsed = JSON.parse(stdout)
          if (parsed.error) {
            reject(new Error(parsed.error))
          } else {
            resolve(parsed as RecommendSpResponse)
          }
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${(e as Error).message}`))
        }
      })

      proc.stdin.write(payload)
      proc.stdin.end()
    })

    // Log inference for overview / activity / charts
    try {
      logInference({
        ffteFeedSolidsSP: Number(body.ffteFeedSolidsSP) || 0,
        ffteProductionSolidsSP: Number(body.ffteProductionSolidsSP) || 0,
        ffteSteamPressureSP: Number(body.ffteSteamPressureSP) || 0,
        tfeOutFlowSP: Number(body.tfeOutFlowSP) || 0,
        tfeProductionSolidsSP: Number(body.tfeProductionSolidsSP) || 0,
        tfeVacuumPressureSP: Number(body.tfeVacuumPressureSP) || 0,
        tfeSteamPressureSP: Number(body.tfeSteamPressureSP) || 0,
        prediction: result.prediction,
        pGood: result.pGood,
        pDowntime: result.pDowntime,
        downtimeRisk: result.downtimeRisk,
        recommended: {
          ffteFeedSolidsSP: result.recommendedSP.ffteFeedSolidsSP,
          ffteProductionSolidsSP: result.recommendedSP.ffteProductionSolidsSP,
          ffteSteamPressureSP: result.recommendedSP.ffteSteamPressureSP,
          tfeOutFlowSP: result.recommendedSP.tfeOutFlowSP,
          tfeProductionSolidsSP: result.recommendedSP.tfeProductionSolidsSP,
          tfeVacuumPressureSP: result.recommendedSP.tfeVacuumPressureSP,
          tfeSteamPressureSP: result.recommendedSP.tfeSteamPressureSP,
        },
      })
    } catch (e) {
      console.error('Failed to log inference:', e)
    }

    return NextResponse.json(result satisfies RecommendSpResponse)
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Recommend SP failed' },
      { status: 500 }
    )
  }
}

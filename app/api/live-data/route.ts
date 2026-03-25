import { NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'

// ── In-memory cursor (resets on server restart) ────────────────────────────
let cursor = 0
let cachedRows: Record<string, string | number>[] | null = null

const DATA_DIR = path.join(process.cwd(), 'data', 'Theme3', 'data_02_07_2019-26-06-2020')

// Column name → camelCase key mapping (matches AllSPInputs)
const SP_MAP: Record<string, string> = {
  'FFTE Feed solids SP':       'ffteFeedSolidsSP',
  'FFTE Production solids SP': 'ffteProductionSolidsSP',
  'FFTE Steam pressure SP':    'ffteSteamPressureSP',
  'TFE Out flow SP':           'tfeOutFlowSP',
  'TFE Production solids SP':  'tfeProductionSolidsSP',
  'TFE Vacuum pressure SP':    'tfeVacuumPressureSP',
  'TFE Steam pressure SP':     'tfeSteamPressureSP',
}

function parseCSV(filePath: string, label: string): Record<string, string | number>[] {
  const text = fs.readFileSync(filePath, 'utf-8')
  const lines = text.split('\n').filter(Boolean)
  const headers = lines[0].split(',').map((h) => h.trim())
  return lines.slice(1).map((line) => {
    const vals = line.split(',')
    const row: Record<string, string | number> = { quality: label }
    headers.forEach((h, i) => {
      const v = vals[i]?.trim() ?? ''
      row[h] = isNaN(Number(v)) || v === '' ? v : Number(v)
    })
    return row
  })
}

function loadData() {
  if (cachedRows) return cachedRows

  const good    = parseCSV(path.join(DATA_DIR, 'good.csv'),     'good')
  const lowBad  = parseCSV(path.join(DATA_DIR, 'low bad.csv'),  'low_bad')
  const highBad = parseCSV(path.join(DATA_DIR, 'high bad.csv'), 'high bad')

  const all = [...good, ...lowBad, ...highBad]

  // Sort by "Set Time" ascending
  all.sort((a, b) => {
    const ta = String(a['Set Time'] ?? '')
    const tb = String(b['Set Time'] ?? '')
    return ta.localeCompare(tb)
  })

  cachedRows = all
  console.log(`[live-data] Loaded ${all.length} rows from CSV`)
  return cachedRows
}

// ── Critical Sensor Mapping (based on NB Feature Importance) ────────────────
const CRITICAL_DRIVERS: Record<string, string[]> = {
  'Yeast - BRN': ['FFTE Feed flow rate PV', 'TFE Out flow SP', 'FFTE Heat temperature 1', 'FFTE Feed solids PV'],
  'Yeast - BRD': ['TFE Out flow SP', 'FFTE Production solids SP', 'FFTE Heat temperature 1', 'FFTE Steam pressure PV'],
  'Yeast - FMX': ['FFTE Heat temperature 3', 'FFTE Discharge density', 'TFE Production solids SP', 'FFTE Heat temperature 1'],
  'default':     ['FFTE Feed flow rate PV', 'FFTE Heat temperature 1', 'FFTE Production solids SP', 'FFTE Discharge density']
}

export async function GET() {
  try {
    const rows = loadData()
    const STEP = 100  // advance 100 rows per tick for faster demo cycling
    const row = rows[cursor % rows.length]
    cursor += STEP

    // Build SP inputs (camelCase keys)
    const sp: Record<string, number | string> = {}
    for (const [csvCol, camelKey] of Object.entries(SP_MAP)) {
      sp[camelKey] = Number(row[csvCol] ?? 0)
    }

    // Part (yeast type)
    const part = String(row['Part'] ?? 'Yeast - BRD')

    // Determine critical sensors for this specific part
    const criticalSensors = CRITICAL_DRIVERS[part] || CRITICAL_DRIVERS['default']

    // Sensor readings (pass-through for display)
    const sensors: Record<string, number> = {}
    const sensorCols = [
      'Extract tank Level', 'FFTE Discharge density', 'FFTE Discharge solids',
      'FFTE Feed flow rate PV', 'FFTE Feed solids PV',
      'FFTE Heat temperature 1', 'FFTE Heat temperature 2', 'FFTE Heat temperature 3',
      'FFTE Production solids PV', 'FFTE Steam pressure PV',
      'TFE Input flow PV', 'TFE Level', 'TFE Motor current', 'TFE Motor speed',
      'TFE Out flow PV', 'TFE Product out temperature', 'TFE Production solids PV',
      'TFE Production solids density', 'TFE Steam pressure PV',
      'TFE Steam temperature', 'TFE Tank level', 'TFE Temperature', 'TFE Vacuum pressure PV',
      'TFE Out flow SP', 'FFTE Production solids SP', 'TFE Production solids SP', // High importance SPs
    ]
    for (const col of sensorCols) {
      sensors[col] = Number(row[col] ?? 0)
    }

    return NextResponse.json({
      cursor,
      total: rows.length,
      timestamp: row['Set Time'],
      part,
      sp,
      sensors,
      criticalSensors,   // NEW: SOTA direction for the dashboard
      quality: row['quality'],   // ground truth label (for display only)
      batch: row['VYP batch'],
    })
  } catch (err) {
    console.error('[live-data] Error:', err)
    return NextResponse.json({ error: String(err) }, { status: 500 })
  }
}

// Reset cursor (for testing)
export async function DELETE() {
  cursor = 0
  return NextResponse.json({ ok: true })
}

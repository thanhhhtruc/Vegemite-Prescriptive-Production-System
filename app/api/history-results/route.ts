import { NextRequest, NextResponse } from 'next/server'
import fs from 'node:fs/promises'
import path from 'node:path'

function parseCsvReverse(content: string, limit = 200) {
  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (lines.length <= 1) {
    return []
  }
  const headers = lines[0].split(',')
  const rawRows = lines.slice(1).reverse().slice(0, limit)
  
  return rawRows.map(line => {
    const vals = line.split(',')
    const obj: any = {}
    headers.forEach((h, i) => {
      obj[h.trim()] = vals[i]
    })
    return obj
  })
}

export async function GET(request: NextRequest) {
  try {
    const absPath = path.join(process.cwd(), 'data/prediction_logs.csv')
    const buf = await fs.readFile(absPath, 'utf8')
    const rows = parseCsvReverse(buf, 500)

    return NextResponse.json({ rows })
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Failed to read history logs' },
      { status: 500 },
    )
  }
}

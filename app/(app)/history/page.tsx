'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { AlertCircle, CheckCircle2, TrendingDown, ChevronLeft, ChevronRight } from 'lucide-react'

// match the columns of prediction_logs.csv
type LogRow = {
  Timestamp: string
  Part: string
  Prediction: string
  Mode?: string
  pGood: string
  pDowntimeRisk: string
  RootCause: string
  rec_pGood?: string
  rec_pDowntimeRisk?: string
  [key: string]: any
}

export default function HistoryPage() {
  const [data, setData] = useState<LogRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const [currentPage, setCurrentPage] = useState(1)
  const rowsPerPage = 20

  const fetchLogs = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/history-results')
      if (!res.ok) throw new Error('Failed to load history logs')
      const json = await res.json()
      setData(json.rows || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error fetching logs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
    
    // auto-refresh every 5s if user stays on this page
    const timer = setInterval(fetchLogs, 5000)
    return () => clearInterval(timer)
  }, [])

  const totalPages = Math.max(1, Math.ceil(data.length / rowsPerPage))
  const paginatedData = data.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage)

  return (
    <main className="flex h-full flex-col gap-4 overflow-hidden p-4 lg:p-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">History Results</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Monitor model predictions, errors detected, and prescriptive risk reduction.
          </p>
        </div>
      </div>

      <Card className="flex min-h-0 flex-1 flex-col">
        <CardHeader className="p-4 pb-2">
          <CardTitle className="text-sm font-semibold tracking-tight">Inference Logs</CardTitle>
          <CardDescription className="text-xs">
            Showing the latest AI predictions sorted by time descending.
          </CardDescription>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 overflow-auto p-0">
          {error ? (
            <div className="p-4 text-xs text-destructive">{error}</div>
          ) : loading && data.length === 0 ? (
            <div className="p-4 text-xs text-muted-foreground">Loading history...</div>
          ) : data.length === 0 ? (
            <div className="p-4 text-xs text-muted-foreground">No prediction logs yet. Please wait for the application to push logs.</div>
          ) : (
            <Table className="w-full min-w-[900px] table-fixed">
              <TableHeader className="sticky top-0 z-10 bg-card shadow-[0_1px_0_0_hsl(var(--border))]">
                <TableRow>
                  <TableHead className="w-[140px]">Time</TableHead>
                  <TableHead className="w-[120px]">Machine / Part</TableHead>
                  <TableHead className="w-[140px]">Prediction Status</TableHead>
                  <TableHead className="w-[200px]">Root Cause / Anomaly</TableHead>
                  <TableHead className="text-right w-[100px]">Original Risk</TableHead>
                  <TableHead className="text-right w-[160px]">Projected Fix</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                  {paginatedData.map((row, idx) => {
                    const isError = row.Prediction === 'HIGH_BAD' || row.Prediction === 'LOW_BAD'
                    const origRisk = parseFloat(row.pDowntimeRisk) || 0
                    let fixRiskLabel = '-'
                    let decreasedBy = 0
                    
                    if (row.rec_pDowntimeRisk) {
                      const fixRisk = parseFloat(row.rec_pDowntimeRisk) || 0
                      decreasedBy = (origRisk - fixRisk)
                      if (decreasedBy < 0) decreasedBy = 0
                      fixRiskLabel = `${fixRisk.toFixed(2)}%`
                    }

                    return (
                      <TableRow key={idx} className="hover:bg-muted/40">
                        <TableCell className="whitespace-nowrap text-xs tabular-nums text-muted-foreground truncate">
                          {row.Timestamp}
                        </TableCell>
                        <TableCell className="whitespace-nowrap text-xs font-medium truncate">
                          <div className="flex items-center gap-1.5">
                            {row.Part}
                            {row.Mode === 'Manual' && (
                              <Badge variant="outline" className="h-4 px-1 text-[9px] uppercase border-purple-200 bg-purple-50 text-purple-600">Manual</Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="truncate">
                          {isError ? (
                            <Badge variant="destructive" className="flex w-max items-center gap-1.5 px-2 py-0.5 text-[10px] font-medium whitespace-nowrap truncate border-destructive/20 bg-destructive/10 text-destructive hover:bg-destructive/20">
                              <AlertCircle className="size-3" />
                              {row.Prediction.replace('_', ' ')}
                            </Badge>
                          ) : (
                            <Badge variant="secondary" className="flex w-max items-center gap-1.5 px-2 py-0.5 text-[10px] font-medium whitespace-nowrap border-emerald-200 bg-emerald-500/10 text-emerald-600 hover:bg-emerald-500/20">
                              <CheckCircle2 className="size-3" />
                              NORMAL
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground truncate" title={isError ? row.RootCause : 'No Issue Detected'}>
                          {isError ? row.RootCause : 'No Issue Detected'}
                        </TableCell>
                        <TableCell className="whitespace-nowrap text-right text-xs font-semibold tabular-nums text-destructive truncate">
                          {isError ? `${origRisk.toFixed(2)}%` : '-'}
                        </TableCell>
                        <TableCell className="whitespace-nowrap text-right text-xs tabular-nums truncate">
                          {isError ? (
                            <div className="flex items-center justify-end gap-2">
                              {row.rec_pDowntimeRisk && decreasedBy > 0 ? (
                                <Badge variant="outline" className="h-5 px-1.5 text-[10px] font-semibold text-emerald-500 border-emerald-200 bg-emerald-500/10" title="Expected Risk Reduction">
                                  <TrendingDown className="mr-1 size-3" />
                                  {decreasedBy.toFixed(2)}%
                                </Badge>
                              ) : null}
                              <span className={decreasedBy > 0 ? "text-emerald-500 font-medium" : "text-muted-foreground"}>
                                {row.rec_pDowntimeRisk ? fixRiskLabel : 'Wait for fix...'}
                              </span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
          )}
        </CardContent>
        {data.length > 0 && (
          <div className="flex items-center justify-between px-4 py-3 border-t bg-muted/20">
            <span className="text-xs text-muted-foreground">
              Showing {Math.min((currentPage - 1) * rowsPerPage + 1, data.length)} to {Math.min(currentPage * rowsPerPage, data.length)} of {data.length} logs
            </span>
            <div className="flex items-center gap-2">
              <Button 
                variant="outline" 
                size="icon" 
                className="h-7 w-7" 
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-xs font-medium tabular-nums min-w-[4ch] text-center">
                {currentPage} / {totalPages}
              </span>
              <Button 
                variant="outline" 
                size="icon" 
                className="h-7 w-7" 
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </Card>
    </main>
  )
}

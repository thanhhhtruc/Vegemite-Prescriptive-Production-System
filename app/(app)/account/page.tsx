'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Mail, Building2, Calendar, Shield } from 'lucide-react'

const MOCK = {
  name: 'Minh Hoang',
  email: 'minh.hoang@vegemite.au',
  role: 'Production Engineer',
  department: 'Paste Production',
  lastLogin: '17 Feb 2026, 09:42',
  memberSince: 'March 2023',
  initials: 'MH',
  avatarUrl: '/minhhoang.jpg',
}

export default function AccountPage() {
  return (
    <main className="p-6 lg:p-12">
      <div className="w-full max-w-3xl mx-auto space-y-12">
        {/* Profile Header Section */}
        <div className="flex flex-col md:flex-row items-center md:items-start gap-8 text-center md:text-left">
          <Avatar className="h-32 w-32 rounded-3xl border-4 border-background shadow-xl">
            {MOCK.avatarUrl && <AvatarImage src={MOCK.avatarUrl} alt={MOCK.name} className="object-cover" />}
            <AvatarFallback className="rounded-3xl text-3xl font-bold bg-primary/10 text-primary">
              {MOCK.initials}
            </AvatarFallback>
          </Avatar>
          
          <div className="flex-1 space-y-3 pt-2">
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
                {MOCK.name}
              </h1>
              <p className="text-base font-medium text-muted-foreground mt-0.5">
                {MOCK.role}
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center md:justify-start gap-3">
              <Badge variant="secondary" className="px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider">
                <Shield className="mr-1 h-3 w-3" />
                System Architect
              </Badge>
              <Badge variant="outline" className="px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider border-primary/20 bg-primary/5 text-primary">
                Administrator
              </Badge>
            </div>
          </div>
        </div>

        <Separator className="bg-border/50" />

        {/* Info Grid */}
        <div className="grid gap-6 sm:grid-cols-2">
          <div className="flex flex-col gap-0.5 p-4 rounded-2xl transition-colors hover:bg-muted/30">
            <div className="flex items-center gap-2 text-muted-foreground mb-0.5">
              <Mail className="h-3.5 w-3.5" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Email Address</span>
            </div>
            <p className="text-base font-semibold">{MOCK.email}</p>
          </div>

          <div className="flex flex-col gap-0.5 p-4 rounded-2xl transition-colors hover:bg-muted/30">
            <div className="flex items-center gap-2 text-muted-foreground mb-0.5">
              <Building2 className="h-3.5 w-3.5" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Department</span>
            </div>
            <p className="text-base font-semibold">{MOCK.department}</p>
          </div>

          <div className="flex flex-col gap-0.5 p-4 rounded-2xl transition-colors hover:bg-muted/30">
            <div className="flex items-center gap-2 text-muted-foreground mb-0.5">
              <Calendar className="h-3.5 w-3.5" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Last Activity</span>
            </div>
            <p className="text-base font-semibold">{MOCK.lastLogin}</p>
          </div>

          <div className="flex flex-col gap-0.5 p-4 rounded-2xl transition-colors hover:bg-muted/30">
            <div className="flex items-center gap-2 text-muted-foreground mb-0.5">
              <Calendar className="h-3.5 w-3.5" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Employee Since</span>
            </div>
            <p className="text-base font-semibold">{MOCK.memberSince}</p>
          </div>
        </div>
        
        {/* Aesthetic footer or subtle note */}
        <div className="pt-12 text-center">
            <p className="text-xs text-muted-foreground/50 font-medium">
                Vegemite Prescriptive Production System · Identity Management
            </p>
        </div>
      </div>
    </main>
  )
}

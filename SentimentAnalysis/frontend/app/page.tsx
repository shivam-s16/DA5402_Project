"use client"

import { useEffect, useState } from "react"
import Image from "next/image"
import Link from "next/link"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tag } from "@/components/tag"
import { ReactionButtons } from "@/components/reaction-buttons"
import { RefreshCw } from "lucide-react"
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs"

interface Article {
  id: number
  headline: string
  pub_date: string
  thumbnail_url: string
  article_url: string
  categories: string[]
  reactions: {
    positive: number
    neutral: number
    negative: number
  }
  sentiment: "positive" | "neutral" | "negative"
}

export default function NewsMoodTabs() {
  const [articles, setArticles] = useState<Article[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchArticles()
  }, [])

  const fetchArticles = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch("/api/news")
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const data = await response.json()
      setArticles(data)
    } catch (error) {
      setError("Network Error: Failed to connect to the API. Displaying sample articles.")
      setArticles([
        {
          id: 1,
          headline: "Positive sample article",
          pub_date: "2025-04-29",
          thumbnail_url: "https://news.google.com/api/attachments/CC8iI0NnNXRjM05GVDJwd1ZVTndlWFpQVFJDeUF4akNCU2dLTWdB=-w280-h168-p-df-rw",
          article_url: "#",
          categories: ["Politics", "World News"],
          reactions: { positive: 12, neutral: 5, negative: 3 },
          sentiment: "positive",
        },
        {
          id: 2,
          headline: "Neutral sample article",
          pub_date: "2025-05-01",
          thumbnail_url: "/placeholder.svg?height=300&width=500",
          article_url: "#",
          categories: ["Technology", "AI", "Business"],
          reactions: { positive: 45, neutral: 8, negative: 2 },
          sentiment: "neutral",
        },
        {
          id: 3,
          headline: "Negative sample article",
          pub_date: "2025-05-02",
          thumbnail_url: "/placeholder.svg?height=300&width=500",
          article_url: "#",
          categories: ["Business", "Economy", "World News"],
          reactions: { positive: 7, neutral: 12, negative: 28 },
          sentiment: "negative",
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  // Group articles by sentiment
  const grouped = {
    positive: articles.filter((a) => a.sentiment === "positive"),
    neutral: articles.filter((a) => a.sentiment === "neutral"),
    negative: articles.filter((a) => a.sentiment === "negative"),
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white py-6 border-b">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center">
            <h1 className="text-4xl font-bold text-teal-600">NewsMood</h1>
            <Button
              onClick={fetchArticles}
              disabled={loading}
              variant="outline"
              className="flex items-center gap-2"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {error && (
          <div className="bg-red-100 text-red-600 p-3 rounded-md text-center mb-8">
            {error}
          </div>
        )}

        <Tabs defaultValue="positive" className="w-full">
          <TabsList className="w-full flex justify-between mb-8 p-1 bg-gray-100 rounded-lg">
            <TabsTrigger
              value="positive"
              className="flex-1 py-4 text-2xl font-bold data-[state=active]:bg-green-200 data-[state=active]:text-green-800 data-[state=active]:shadow-lg transition-all rounded-lg"
            >
              Positive
            </TabsTrigger>
            <TabsTrigger
              value="neutral"
              className="flex-1 py-4 text-2xl font-bold data-[state=active]:bg-blue-200 data-[state=active]:text-blue-800 data-[state=active]:shadow-lg transition-all rounded-lg"
            >
              Neutral
            </TabsTrigger>
            <TabsTrigger
              value="negative"
              className="flex-1 py-4 text-2xl font-bold data-[state=active]:bg-red-200 data-[state=active]:text-red-800 data-[state=active]:shadow-lg transition-all rounded-lg"
            >
              Negative
            </TabsTrigger>
          </TabsList>

          <TabsContent value="positive">
            {loading ? (
              <LoadingGrid />
            ) : grouped.positive.length ? (
              <NewsGrid articles={grouped.positive} />
            ) : (
              <EmptyState sentiment="positive" />
            )}
          </TabsContent>
          <TabsContent value="neutral">
            {loading ? (
              <LoadingGrid />
            ) : grouped.neutral.length ? (
              <NewsGrid articles={grouped.neutral} />
            ) : (
              <EmptyState sentiment="neutral" />
            )}
          </TabsContent>
          <TabsContent value="negative">
            {loading ? (
              <LoadingGrid />
            ) : grouped.negative.length ? (
              <NewsGrid articles={grouped.negative} />
            ) : (
              <EmptyState sentiment="negative" />
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

// Helper: News card grid
function NewsGrid({ articles }: { articles: Article[] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {articles.map((article) => (
        <Card key={article.id} className="overflow-hidden">
          <div className="aspect-[16/9] relative">
            <Image
              src={article.thumbnail_url || "/placeholder.svg?height=300&width=500"}
              alt={article.headline}
              fill
              className="object-cover"
              sizes="(max-width: 768px) 100vw, 33vw"
            />
          </div>
          <div className="p-5">
            <h3 className="text-xl font-bold mb-3">
              <Link href={article.article_url} className="hover:text-teal-600 transition-colors">
                {article.headline}
              </Link>
            </h3>
            <p className="text-gray-500 mb-4">Published: {article.pub_date}</p>
            <div className="flex flex-wrap gap-2 mb-4">
              {article.categories.map((category) => (
                <Tag key={category} label={category} />
              ))}
            </div>
          </div>
          <div className="border-t p-4">
            <ReactionButtons articleId={article.id} initialReactions={article.reactions} />
          </div>
        </Card>
      ))}
    </div>
  )
}

// Helper: Loading skeleton grid
function LoadingGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {[...Array(3)].map((_, idx) => (
        <Card key={idx} className="p-5">
          <Skeleton className="aspect-[16/9] w-full mb-4" />
          <Skeleton className="h-6 w-3/4 mb-2" />
          <Skeleton className="h-4 w-1/2 mb-2" />
          <div className="flex gap-2 mb-4">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-5 w-16" />
          </div>
          <Skeleton className="h-8 w-full" />
        </Card>
      ))}
    </div>
  )
}

// Helper: Empty state
function EmptyState({ sentiment }: { sentiment: string }) {
  return (
    <div className="text-center text-gray-400 py-12">
      No {sentiment} news found for this date range.
    </div>
  )
}

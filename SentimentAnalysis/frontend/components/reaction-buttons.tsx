"use client"

import { useState } from "react"
import { ThumbsUp, ThumbsDown, Smile } from "lucide-react"
import { cn } from "@/lib/utils"

type SentimentType = "positive" | "neutral" | "negative"

interface ReactionButtonsProps {
  articleId: number
  initialReactions?: {
    positive: number
    neutral: number
    negative: number
  }
}

export function ReactionButtons({
  articleId,
  initialReactions = { positive: 0, neutral: 0, negative: 0 },
}: ReactionButtonsProps) {
  const [selectedReaction, setSelectedReaction] = useState<SentimentType | null>(null)
  const [reactions, setReactions] = useState(initialReactions)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleReaction = async (sentiment: SentimentType) => {
    if (isSubmitting) return

    setIsSubmitting(true)

    // If clicking the same reaction, deselect it
    if (selectedReaction === sentiment) {
      setSelectedReaction(null)
      // Decrease the count for the deselected reaction
      setReactions((prev) => ({
        ...prev,
        [sentiment]: Math.max(0, prev[sentiment] - 1),
      }))
    } else {
      // If changing reaction, decrease previous and increase new
      if (selectedReaction) {
        setReactions((prev) => ({
          ...prev,
          [selectedReaction]: Math.max(0, prev[selectedReaction] - 1),
          [sentiment]: prev[sentiment] + 1,
        }))
      } else {
        // Just increase the new reaction
        setReactions((prev) => ({
          ...prev,
          [sentiment]: prev[sentiment] + 1,
        }))
      }
      setSelectedReaction(sentiment)
    }

    try {
      // Send to API
      await fetch("/api/reaction", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          articleId,
          sentiment,
          emoji: sentiment === "positive" ? "👍" : sentiment === "neutral" ? "😐" : "👎",
        }),
      })
    } catch (error) {
      console.error("Error submitting reaction:", error)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="flex justify-between items-center">
      <button
        onClick={() => handleReaction("positive")}
        className={cn(
          "flex items-center gap-2 p-2 rounded-md transition-colors",
          selectedReaction === "positive" ? "text-teal-600" : "text-gray-500 hover:text-teal-600",
        )}
        disabled={isSubmitting}
        aria-label="Positive reaction"
      >
        <ThumbsUp className="h-5 w-5" />
        {reactions.positive > 0 && <span className="text-sm">{reactions.positive}</span>}
      </button>

      <button
        onClick={() => handleReaction("neutral")}
        className={cn(
          "flex items-center gap-2 p-2 rounded-md transition-colors",
          selectedReaction === "neutral" ? "text-amber-500" : "text-gray-500 hover:text-amber-500",
        )}
        disabled={isSubmitting}
        aria-label="Neutral reaction"
      >
        <Smile className="h-5 w-5" />
        {reactions.neutral > 0 && <span className="text-sm">{reactions.neutral}</span>}
      </button>

      <button
        onClick={() => handleReaction("negative")}
        className={cn(
          "flex items-center gap-2 p-2 rounded-md transition-colors",
          selectedReaction === "negative" ? "text-red-500" : "text-gray-500 hover:text-red-500",
        )}
        disabled={isSubmitting}
        aria-label="Negative reaction"
      >
        <ThumbsDown className="h-5 w-5" />
        {reactions.negative > 0 && <span className="text-sm">{reactions.negative}</span>}
      </button>
    </div>
  )
}

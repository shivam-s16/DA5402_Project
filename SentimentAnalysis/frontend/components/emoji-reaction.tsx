"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"

type EmojiType = {
  symbol: string
  sentiment: "positive" | "negative" | "neutral"
  label: string
}

const emojis: EmojiType[] = [
  { symbol: "👍", sentiment: "positive", label: "Like" },
  { symbol: "❤️", sentiment: "positive", label: "Love" },
  { symbol: "😂", sentiment: "positive", label: "Laugh" },
  { symbol: "😐", sentiment: "neutral", label: "Neutral" },
  { symbol: "😡", sentiment: "negative", label: "Angry" },
  { symbol: "👎", sentiment: "negative", label: "Dislike" },
]

interface EmojiReactionProps {
  articleId: number
}

export function EmojiReaction({ articleId }: EmojiReactionProps) {
  const [selectedEmoji, setSelectedEmoji] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const handleEmojiClick = async (emoji: EmojiType) => {
    try {
      setIsSubmitting(true)
      setSelectedEmoji(emoji.symbol)
      setMessage(null)

      const response = await fetch("/api/reaction", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          articleId,
          sentiment: emoji.sentiment,
          emoji: emoji.symbol,
        }),
      })

      const data = await response.json()

      if (data.success) {
        setMessage(`Thanks for your ${emoji.sentiment} reaction!`)
      } else {
        setMessage("Failed to record your reaction. Please try again.")
        setSelectedEmoji(null)
      }
    } catch (error) {
      console.error("Error submitting reaction:", error)
      setMessage("An error occurred. Please try again.")
      setSelectedEmoji(null)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="mt-4">
      <p className="text-sm text-gray-500 mb-2">How did this article make you feel?</p>
      <div className="flex flex-wrap gap-2">
        {emojis.map((emoji) => (
          <button
            key={emoji.symbol}
            onClick={() => handleEmojiClick(emoji)}
            disabled={isSubmitting}
            className={cn(
              "text-xl p-2 rounded-full transition-all hover:bg-gray-100 hover:scale-110",
              selectedEmoji === emoji.symbol && "bg-gray-200 scale-110",
              isSubmitting && "opacity-50 cursor-not-allowed",
            )}
            title={emoji.label}
            aria-label={emoji.label}
          >
            <span role="img" aria-label={emoji.label}>
              {emoji.symbol}
            </span>
          </button>
        ))}
      </div>
      {message && (
        <p className={cn("text-sm mt-2", message.includes("Thanks") ? "text-green-600" : "text-red-600")}>{message}</p>
      )}
    </div>
  )
}

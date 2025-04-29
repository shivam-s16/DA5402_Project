import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { articleId, sentiment, emoji } = body;

    // Log the reaction data
    console.log(`Article ID: ${articleId}, Sentiment: ${sentiment}, Emoji: ${emoji}`);

    // Map sentiment to feedback_type (assuming positive/negative/neutral)
    const feedback_type = sentiment; // Adjust if your backend expects different values

    // Construct the backend URL
    const backendUrl = `http://localhost:8000/articles/${articleId}/feedback/?feedback_type=${feedback_type}`;

    // Forward the reaction to the backend (POST request)
    const backendRes = await fetch(backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ emoji }), // Send emoji if your backend expects it
    });

    // Check backend response
    if (!backendRes.ok) {
      throw new Error("Backend failed to record feedback");
    }

    // Optionally, you can parse backend response
    const backendData = await backendRes.json();

    return NextResponse.json({
      success: true,
      message: `Reaction ${emoji} (${sentiment}) recorded for article ${articleId}`,
      backend: backendData,
    });
  } catch (error) {
    console.error("Error processing reaction:", error);
    return NextResponse.json(
      { success: false, message: "Failed to process reaction" },
      { status: 500 }
    );
  }
}

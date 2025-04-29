import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  // Get start_date and end_date from query params
  let start_date = searchParams.get("start_date");
  let end_date = searchParams.get("end_date");

  // If not provided, use today's and tomorrow's date (YYYY-MM-DD)
  if (!start_date || !end_date) {
    const today = new Date();
    const tomorrow = new Date();
    tomorrow.setDate(today.getDate() + 1);

    // Format to YYYY-MM-DD
    const pad = (n: number) => n.toString().padStart(2, "0");
    const format = (d: Date) =>
      `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

    start_date = format(today);
    end_date = format(tomorrow);
  }

  const backendUrl = `http://localhost:8000/articles/?start_date=${start_date}&end_date=${end_date}`;
  const res = await fetch(backendUrl);
  const articles = await res.json();

  return NextResponse.json(articles);
}

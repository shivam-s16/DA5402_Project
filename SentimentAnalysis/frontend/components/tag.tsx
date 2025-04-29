interface TagProps {
  label: string
}

export function Tag({ label }: TagProps) {
  return (
    <span className="inline-block bg-gray-100 text-gray-800 px-3 py-1 text-sm font-medium rounded-md">{label}</span>
  )
}

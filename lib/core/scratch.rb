require_relative 'Utilities'

floorToPreventOverflow = 1e-30
largestEuclidianDistanceMoved = 1e35

return1 =unless (largestEuclidianDistanceMoved.nil?)
  largestEuclidianDistanceMoved
else
  floorToPreventOverflow
end


return2 = largestEuclidianDistanceMoved ||= floorToPreventOverflow

puts return1, return2
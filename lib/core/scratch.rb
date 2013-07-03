require_relative 'Utilities'

include Math


def correctionFactorForRateAtWhichNeuronsGainChanges(clustersCenter, correctionFactorsFloor = 0.1)
  c = clustersCenter
  m = (exp(-c) + 1)

  n = m**2
  o = (2.0 * exp(-2.0 * c)) / (m**3)
  p = -1.0 * (exp(-c) / m**2)
  q = exp(c)

  result = (n * (o + p) * q).abs
  return [result, correctionFactorsFloor].max
end

clustersCenter = -1.0
correctionFactorFloor = 0.1

result = correctionFactorForRateAtWhichNeuronsGainChanges(clustersCenter, correctionFactorFloor)

puts "result=\t#{result}"

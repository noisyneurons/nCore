#require_relative 'Utilities'

def rFalse
  return false
end

def findCommonLink(outputLinks, inputLinks)
  theCommonLink = outputLinks.find do |anOutputLink|
    inputLinks.find { |anInputLink| anInputLink == anOutputLink }
  end
end

outputLinks = [1, 2, 3, 4, 5]
inputLinks = [6, 7, 225, 21]

puts "result of search=\t #{findCommonLink(outputLinks, inputLinks)}"

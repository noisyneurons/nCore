#FunctionalTestOfMixtureOfModels.rb

require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require 'distribution'



def suppress?(aValue, reverse)
  returnValue = aValue

  case reverse
    when false
      returnValue
    when true
      !returnValue
  end
end

reverse = false
aValue = true

puts suppress?(aValue, reverse)
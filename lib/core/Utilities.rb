# Utilities.rb

require 'rubygems'
require 'bundler/setup'
require 'stringio'
require 'mathn'
require 'matrix'
require 'redis'
require 'yaml'

# Globals, Constants
INFINITY = 1.0/0

localhostIP = "127.0.0.1"
remoteAlwaysOnIP = "54.164.134.152"
rediscontainerDomainName = "db"
localRediscontainerIP = "172.17.0.2"

# $currentHost = 'db'
# $currentHost = localRediscontainerIP
$currentHost = remoteAlwaysOnIP
# $currentHost = localhostIP
# $currentHost = rediscontainerDomainName

$redis = Redis.new(:host => $currentHost, :password => "com")

############################# MODULES ###########################

module OS
  def OS.windows?
    (/cygwin|mswin|mingw|bccwin|wince|emx/ =~ RUBY_PLATFORM) != nil
  end

  def OS.mac?
    (/darwin/ =~ RUBY_PLATFORM) != nil
  end

  def OS.unix?
    !OS.windows?
  end

  def OS.linux?
    OS.unix? and not OS.mac?
  end
end

############################# DEBUG UTILITY FUNCTIONS ###########################

def std(txt, x)
  STDOUT.logger.puts "#{txt}\t#{x}"; STDOUT.flush
end

def qreport(dataArray, epochNumber, interval)
  if ((epochNumber-1).modulo(interval) == 0)
    STDOUT.print "Epoch Number=\t#{epochNumber} --\t"
    dataArray.each_with_index do |dataItem, indexToDataItem|
      STDOUT.print "dataItem #{indexToDataItem} =\t#{dataItem};\t"
    end
    STDOUT.flush
    STDOUT.logger.puts
  end
end

def periodicallyDisplayContentsOfHash(hashWithData, epochNumber, interval)
  if ((epochNumber-1).modulo(interval) == 0)
    STDOUT.print "Epoch Number=\t#{epochNumber} -->\t"
    hashWithData.each do |key, value|
      STDOUT.print "#{key} =\t#{value};\t"
    end
    STDOUT.flush
    STDOUT.logger.puts
  end
end


#TODO Subclass Vector and add these methods to the subclass
class Vector
  # Calculates the distance to Point p
  def dist_to(p)
    return (self - p).r
  end
end

#	##########################  Array Extensions ##################
class Array

  def to_LayerAry
    return LayerArray.new(self)
  end

  def to_Layer
    return Layer.new(self)
  end

  def mean
    sumOfArray = self.inject { |sum, n| sum + n }
    return (sumOfArray.to_f / self.length)
  end

  def abs
    self.collect { |value| value.abs }
  end

  def standardError
    meanOfArray = self.mean
    sumOfSquares = self.inject { |sum, n| sum + ((n-meanOfArray)**2) }
    return Math.sqrt(sumOfSquares / self.length)
  end

  def normalize
    meanValue = self.mean
    centeredArray = self.collect { |value| value - meanValue }
    largestAbsoluteArrayElement = centeredArray.minmax.abs.max.to_f
    centeredArray.collect { |value| value / largestAbsoluteArrayElement }
  end

  def scaleValuesToSumToOne
    sumOfArray = self.inject(0.0) { |sum, value| sum + value }
    self.collect { |value| value / sumOfArray }
  end
end

#	##########################  Object Extensions ##################
class Object
  def blank?
    respond_to?(:empty?) ? empty? : !self
  end

  def present?
    !blank?
  end

  def deep_clone
    return @deep_cloning_obj if @deep_cloning
    @deep_cloning_obj = clone
    @deep_cloning_obj.instance_variables.each do |var|
      val = @deep_cloning_obj.instance_variable_get(var)
      begin
        @deep_cloning = true
        val = val.deep_clone
      rescue TypeError
        next
      ensure
        @deep_cloning = false
      end
      @deep_cloning_obj.instance_variable_set(var, val)
    end
    deep_cloning_obj = @deep_cloning_obj
    @deep_cloning_obj = nil
    deep_cloning_obj
  end
end

require 'rubygems'
require 'bundler/setup'
require_relative 'Utilities'

## require '/home/mark/usr/local/ruby2.1.3/ruby/lib/ruby/2.1.0/forwardable'
#require 'logger'
#
require 'stringio'
#require 'logger'
#stringFile = StringIO.new
#logger = Logger.new(stringFile)
#
##logger = Logger.new('test.log')
#
##logger = Logger.new($stdout)
#
##logger = Logger.new(stringFile).tap do |log|
##  log.progname = 'MarksProg'
##end
#
#
#logger.info 'doing some stuff'
#logger.info 'doing some stuff2'
#
#stringFile.logger.puts "something AFTER"
#
#logger.puts stringFile.string
#
##p stringFile.inspect
##log = stringFile.rewind
##logger.puts log


module Logger
  # attr_accessor :logger

  def logger=(aLogger)
    @logger = aLogger
  end

  def logger
    @logger
  end

end

class MyClass
  include Logger
  def initialize
    # loggerSet(StringIO.new)
    self.logger = StringIO.new
  end

  def start
    logger.puts " Hi There"
  end

  def endProgram
    logger.puts logger.string
  end
end

myClass = MyClass.new

myClass.start
myClass.start

myClass.endProgram
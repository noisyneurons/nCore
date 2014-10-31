require 'rubygems'
require 'bundler/setup'


#require_relative 'Utilities'
#require_relative 'DataSet'
#require_relative 'NeuralIOFunctions'



require 'statsample'

# require '/home/mark/usr/local/ruby2.1.3/ruby/lib/ruby/gems/2.1.0/gems/statsample'
# Note R like generation of random gaussian variable
# and correlation matrix

ss_analysis("Statsample::Bivariate.correlation_matrix") do
  samples=1000
  ds=data_frame(
      'a'=>rnorm(samples),
      'b'=>rnorm(samples),
      'c'=>rnorm(samples),
      'd'=>rnorm(samples))
  cm=cor(ds)
  summary(cm)
end

Statsample::Analysis.run_batch # Echo output to console

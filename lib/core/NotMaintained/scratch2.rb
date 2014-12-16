require 'rubygems'

require 'active_record'

ActiveRecord::Base.establish_connection(
    :adapter => "sqlite3",
    :database => ":memory:"
#:database => "../../../data/acrossEpochs.db"
)

class CreateEpochResponses < ActiveRecord::Migration
  def change
    create_table :epoch_responses do |t|
      t.integer :epochNumber
      t.float :cluster0Center
      t.float :cluster1Center
      t.float :dPrime
    end

    create_table :example_responses do |t|
      t.integer :epochNumber
      t.integer :exampleNumber
      t.integer :neuronID
      t.float :netInput
      t.float :error
      t.float :bpError
      t.float :localFlockingError
      t.float :wt1
      t.float :wt2
      t.float :cluster0Center
      t.float :cluster1Center
      t.float :dPrime
      t.float :weightedErrorMetric
      t.references :epoch_response
    end

    create_table :learning_types do |t|
      t.integer :epochNumber
      t.string :learningPhase
      t.references :epoch_response
    end
    add_index :example_responses, :epoch_response_id

  end
end


CreateEpochResponses.new.change


class Epoch_response < ActiveRecord::Base
  attr_accessible :epochNumber, :cluster0Center, :cluster1Center, :dPrime
  has_many :example_responses

  def to_s
    aString = "\n#{id}\t#{epochNumber}\t#{cluster0Center}"
    #example_responses.each do | r |
    #  aString += "\n#{r.exampleNumber}"
    #end
  end
end

class Example_response < ActiveRecord::Base
  attr_accessible :epochNumber, :exampleNumber, :neuronID, :netInput, :error, :bpError,
                  :localFlockingError, :wt1, :wt2, :weightedErrorMetric, :epoch_response_id
  belongs_to :epoch_response

  def to_s
    aString = "\n#{id}\t#{epoch_response_id}\t#{epochNumber}\t#{exampleNumber}"
  end
end

(er = Epoch_response.new(:epochNumber => 1, :cluster0Center => 0.1)).save
aValueInHashNotation = {:epochNumber => 1, :exampleNumber => 3, :epoch_response_id => er.id}
Example_response.new(:epochNumber => 1, :exampleNumber => 1, :epoch_response_id => er.id).save
Example_response.new(:epochNumber => 1, :exampleNumber => 2, :epoch_response_id => er.id).save
Example_response.new(aValueInHashNotation).save
Example_response.new(:epochNumber => 1, :exampleNumber => 4, :epoch_response_id => er.id).save

(er = Epoch_response.new(:epochNumber => 2, :cluster0Center => 0.1)).save
Example_response.new(:epochNumber => 2, :exampleNumber => 1, :epoch_response_id => er.id).save
Example_response.new(:epochNumber => 2, :exampleNumber => 2, :epoch_response_id => er.id).save
Example_response.new(:epochNumber => 2, :exampleNumber => 2, :epoch_response_id => er.id).save
Example_response.new(:epochNumber => 2, :exampleNumber => 4, :epoch_response_id => er.id).save

logger.puts
logger.puts Epoch_response.all
logger.puts
logger.puts Example_response.all

#logger.puts
#Epoch_response.joins(:example_responses)
#logger.puts
#logger.puts Epoch_response.all


#Epoch_response.find_each do |epochResponse|
#
#  logger.puts "000\t#{Example_response.where(:epoch_response_id => epochResponse.id)}"
#
#end

anEpochRecord = Epoch_response.first

logger.puts anEpochRecord
logger.puts
anEpochRecord.example_responses.each { |r| logger.puts r }
# Dockerfile used to build noisyneurons/nn docker image
# sudo docker build -t noisyneurons/nn  .

# Pull base image.
FROM litaio/ruby:2.1.3

RUN apt-get update && \
apt-get install -y ruby-bundler && \
rm -rf /var/lib/apt/lists/*

RUN gem install bundler

# Set gem info for bundler
ADD Gemfile /NN2012/
ADD Gemfile.lock /NN2012/
WORKDIR /NN2012

# use bundler to install appropriate gems
RUN bundle install

CMD ["bash"]









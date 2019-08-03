const tf = require('@tensorflow/tfjs')

class LinearRegression {
  constructor (features, labels, options) {
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels)
    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 1000
    }, options)

    this.weights = tf.zeros([2, 1])
  }

  gradientDescent () {
    const currentGuesses = this.features.matMul(this.weights)
    const differences = currentGuesses.sub(this.labels)

    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0])

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
  }

  train () {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }

  test (features, labels) {
    features = this.processFeatures(features)
    labels = tf.tensor(labels)

    const predictions = features.matMul(this.weights)

    const res = labels
      .sub(predictions)
      .pow(2)
      .sum()
      .get()

    const tot = labels
      .sub(labels.mean())
      .pow(2)
      .sum()
      .get()

    return 1 - res / tot
  }

  processFeatures (features) {
    features = tf.tensor(features)

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5))
    } else {
      this.standardize(features)
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1)

    return features
  }

  standardize (features) {
    const { mean, variance } = tf.moments(features, 0)
    this.mean = mean
    this.variance = variance

    return features.sub(mean).div(variance.pow(0.5))
  }
}

module.exports = LinearRegression

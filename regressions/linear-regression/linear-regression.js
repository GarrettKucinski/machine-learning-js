const tf = require('@tensorflow/tfjs')

class LinearRegression {
  constructor (features, labels, options) {
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels)
    this.mseHistory = []
    this.options = {
      learningRate: 0.1,
      iterations: 1000,
      ...options
    }

    this.weights = tf.zeros([this.features.shape[1], 1])
  }

  gradientDescent (features, labels) {
    const currentGuesses = features.matMul(this.weights)
    const differences = currentGuesses.sub(labels)

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0])

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
  }

  train () {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    )

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const { batchSize } = this.options
        const startIndex = j * batchSize

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        )

        const labelSlice = this.labels.slice(
          [startIndex, 0],
          [batchSize, -1]
        )

        this.gradientDescent(featureSlice, labelSlice)
      }
      this.recordMSE()
      this.updateLearningRate()
    }
  }

  predict (observations) {
    return this.processFeatures(observations).matMul(this.weights)
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
      features = this.standardize(features)
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

  recordMSE () {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get()

    this.mseHistory.unshift(mse)
  }

  updateLearningRate () {
    if (this.mseHistory.length < 2) {
      return
    }

    const [a, b] = this.mseHistory

    if (a > b) {
      this.options.learningRate /= 2
    } else {
      this.options.learningRate *= 1.05
    }
  }
}

module.exports = LinearRegression

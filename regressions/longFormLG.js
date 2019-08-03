const R = require('ramda')

class LongFormLinearRegression {
  constructor (features, labels, options) {
    this.features = features
    this.labels = labels
    this.options = {
      learningRate: 0.001,
      iterations: 1000,
      ...options
    }

    this.m = 0
    this.b = 0
  }

  gradientDescent () {
    const currentGuessesForMpg = this.features.map(([x]) => {
      return this.m * x + this.b
    })

    const bSlope = (R.sum(currentGuessesForMpg.map((guess, i) => {
      return guess - this.labels[i][0]
    })) * 2) / this.features.length

    const mSlope = (R.sum(currentGuessesForMpg.map((guess, i) => {
      return -1 * this.features[i][0] * (this.labels[i][0] - guess)
    })) * 2) / this.features.length

    this.m = this.m - mSlope * this.options.learningRate
    this.b = this.b - bSlope * this.options.learningRate
  }

  train () {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }
}

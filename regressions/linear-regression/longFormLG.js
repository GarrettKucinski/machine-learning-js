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
    /**
     * Guessing a starting value for both the slope (m)
     * and the y-intercept (b) we plug in all of the values
     * for our feature set, one at a time for (x). This gives
     * us an array of guessed (y) values.
     */
    const currentGuessesForMpg = this.features.map(([x]) => {
      return this.m * x + this.b
    })

    /**
     * Take all guessed (y) values and run them through this equation
     * (sum for n guessed values (y) (guessedValueN - actualValueY) ** 2) / totalNumberOfValues
     */
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

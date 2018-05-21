/*
 * ssm.scala
 * author:  Rui Vieira <rui@redhat.com>
 *
 * Copyright (c) 2018 Red Hat, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.radanalytics.silex.ssm

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Gaussian, MultivariateGaussian}

case class Structure(F: DenseMatrix[Double],
                     G: DenseMatrix[Double],
                     W: DenseMatrix[Double]) {
  def +(that: Structure): Structure = {
    val _F = DenseMatrix.vertcat[Double](F, that.F)
    val _G = Utils.bdiag(Seq(G, that.G))
    val _W = Utils.bdiag(Seq(W, that.W))
    Structure(F = _F, G = _G, W = _W)
  }
}

object Utils {
  def bdiag(matrices: Seq[DenseMatrix[Double]]): DenseMatrix[Double] = {
    val dim = matrices.map(_.cols)
    val size = matrices.length

    DenseMatrix.vertcat((0 until size) map { i =>
      DenseMatrix.horzcat((0 until size) map { n =>
        if (i == n) matrices(n)
        else DenseMatrix.zeros[Double](dim(i), dim(n))
      }: _*)
    }: _*)

  }
}

object Structure {
  def locallyConstant(W: Double): Structure = {
    Structure(F = DenseMatrix.eye[Double](dim = 1),
      G = DenseMatrix.eye[Double](dim = 1),
      W =
        DenseMatrix.create[Double](rows = 1, cols = 1, data = Array(W)))
  }

  def locallyLinear(W: DenseMatrix[Double]): Structure = {
    Structure(F = DenseMatrix
      .create[Double](rows = 2, cols = 1, data = Array(1.0, 0.0)),
      G = DenseMatrix.create[Double](rows = 2,
        cols = 2,
        data = Array(1.0, 0.0, 1.0, 1.0)),
      W = W)
  }

  def cyclicFourier(period: Int,
                    harmonics: Int,
                    W: DenseMatrix[Double]): Structure = {
    val omega = 2.0 * Math.PI / period.toFloat
    val harmonic1: DenseMatrix[Double] = DenseMatrix.eye[Double](dim = 2) * Math.cos(omega) // build main harmonic

    harmonic1(0, 1) = Math.sin(omega)
    harmonic1(1, 0) = -harmonic1(0, 1)
    val G = if (harmonics > 1) {
      val h = new Array[DenseMatrix[Double]](harmonics)
      h(0) = harmonic1.copy
      (1 until harmonics) foreach { i =>
        h(i) = h(i - 1) * harmonic1
      }
      Utils.bdiag(h)

    } else {
      harmonic1
    }
    val dim = G.rows
    val F = DenseMatrix.zeros[Double](dim, 1)
    F(::, 0) := new DenseVector[Double](
      Array.fill(harmonics)(Array(1.0, 0.0)).flatten)
    Structure(F = F, G = G, W = W)
  }

}

abstract class DGLM[U](structure: Structure) {

  def nextState(state: DenseVector[Double]): DenseVector[Double] = {
    MultivariateGaussian(structure.G * state, structure.W).sample()
  }

  def observation(state: DenseVector[Double]): U

}

class GaussianDLM(structure: Structure, V: Double)
  extends DGLM[Double](structure = structure) {

  override def observation(state: DenseVector[Double]): Double = {

    new Gaussian((structure.F.t * state).apply(0), V).sample()

  }
}

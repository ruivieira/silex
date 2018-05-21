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
import org.scalatest._

class SsmStructureSpec extends FlatSpec with Matchers {

  def dimensions(structure: Structure, n: Int): Unit = {
    assert(structure.F.cols.equals(1))
    assert(structure.F.rows.equals(n))
    assert(structure.G.cols.equals(n))
    assert(structure.G.rows.equals(n))
    assert(structure.W.cols.equals(n))
    assert(structure.W.rows.equals(n))
  }

  it should "create a locally constant structure with the correct dimensions" in {
    val structure = Structure.locallyConstant(1.0)
    dimensions(structure, 1)
  }
  it should "create a locally linear structure with the correct dimensions" in {
    val structure = Structure.locallyLinear(DenseMatrix.eye[Double](2))
    dimensions(structure, 2)
  }
  it should "create a fourier structure (1 harmonic) with the correct dimensions" in {
    val structure = Structure.cyclicFourier(period = 100,
      harmonics = 1, W = DenseMatrix.eye[Double](2))
    dimensions(structure, 2)
  }
  it should "create a fourier structure (2 harmonics) with the correct dimensions" in {
    val structure = Structure.cyclicFourier(period = 100,
      harmonics = 2, W = DenseMatrix.eye[Double](4))
    dimensions(structure, 4)
  }
  it should "create a fourier structure (3 harmonics) with the correct dimensions" in {
    val structure = Structure.cyclicFourier(period = 100,
      harmonics = 3, W = DenseMatrix.eye[Double](6))
    dimensions(structure, 6)
  }
  it should "compose a locally constant and fourier (1 harmonic) with the correct dimensions" in {
    val lc = Structure.locallyConstant(1.0)
    val fourier = Structure.cyclicFourier(period = 100,
      harmonics = 1, W = DenseMatrix.eye[Double](2))
    val structure = lc + fourier
    dimensions(structure, 3)
  }
  it should "compose two locally constant with the correct dimensions" in {
    val lc = Structure.locallyConstant(1.0)
    val structure = lc + lc
    dimensions(structure, 2)
  }


}

class GaussianDlmSpec extends FlatSpec with Matchers {
  it should "keep state dimensions on propagation with locally constant structure" in {
    val structure = Structure.locallyConstant(1.0)
    val gaussianDLM = new GaussianDLM(structure = structure, V = 1.0)
    val prior = DenseVector.zeros[Double](1)
    val nextState = gaussianDLM.nextState(prior)

    assert(nextState.length.equals(prior.length))
  }
  it should "keep state dimensions on propagation with locally linear structure" in {
    val structure = Structure.locallyLinear(DenseMatrix.eye[Double](2))
    val gaussianDLM = new GaussianDLM(structure = structure, V = 1.0)
    val prior = DenseVector.zeros[Double](2)
    val nextState = gaussianDLM.nextState(prior)

    assert(nextState.length.equals(prior.length))
  }
  it should "keep state dimensions on propagation with fourier (1 harmonic) structure" in {
    val structure = Structure.cyclicFourier(period = 100, harmonics = 1, W = DenseMatrix.eye[Double](2))
    val gaussianDLM = new GaussianDLM(structure = structure, V = 1.0)
    val prior = DenseVector.zeros[Double](2)
    val nextState = gaussianDLM.nextState(prior)

    assert(nextState.length.equals(prior.length))
  }
  it should "keep state dimensions on propagation with fourier (2 harmonic) structure" in {
    val structure = Structure.cyclicFourier(period = 100, harmonics = 2, W = DenseMatrix.eye[Double](4))
    val gaussianDLM = new GaussianDLM(structure = structure, V = 1.0)
    val prior = DenseVector.zeros[Double](4)
    val nextState = gaussianDLM.nextState(prior)

    assert(nextState.length.equals(prior.length))
  }
  it should "keep state dimensions on propagation with fourier (3 harmonic) structure" in {
    val structure = Structure.cyclicFourier(period = 100, harmonics = 3, W = DenseMatrix.eye[Double](6))
    val gaussianDLM = new GaussianDLM(structure = structure, V = 1.0)
    val prior = DenseVector.zeros[Double](6)
    val nextState = gaussianDLM.nextState(prior)

    assert(nextState.length.equals(prior.length))
  }

}
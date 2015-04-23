/*
 * This file is part of the "silex" library of helpers for Apache Spark.
 *
 * Copyright (c) 2015 Red Hat, Inc.
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
 * limitations under the License.c
 */

package com.redhat.et.silex

/** Enriched RDD methods for histogramming and counting of sequence-like objects
  * {{{
  * import com.redhat.et.silex.histogram.implicits._
  * rdd.countBy(f)
  * rdd.histBy(f)
  * rdd.countByFlat(f)
  * rdd.histByFlat(f)
  * }}}
  * @note Currently Spark RDDs are supported, however Scala collections or traversables are
  * also planned for future inclusion
  */
package object histogram {}

/**
 * @file  AISAttributes.h
 * @brief C++ bridge between the generated attribute classes and the rest of the compiler.
 *
 * This header only re-exports what TableGen produced:
 *   - Enum attribute kinds
 *   - Singleton attribute classes
 *
 */

#ifndef APXM_AIS_ATTRIBUTES_H
#define APXM_AIS_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#include "apxm/Dialect/AIS/IR/AISEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "apxm/Dialect/AIS/IR/AISAttributes.h.inc"

#endif  // APXM_AIS_ATTRIBUTES_H

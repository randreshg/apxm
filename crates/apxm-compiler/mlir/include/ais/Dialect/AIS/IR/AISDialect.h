/**
 * @file  AISDialect.h
 * @brief Public entry point for the AIS dialect.
 *
 * Includes the auto-generated dialect declaration (`AISDialect.h.inc`)
 * and pulls in the operation and type headers so that a single `#include`
 * is enough for most translation units.  Keep this header minimal: any
 * inline helper that is not universally needed should go into a separate
 * *Utils.h file to avoid unnecessary rebuilds.
 */

#ifndef APXM_AIS_DIALECT_H
#define APXM_AIS_DIALECT_H

#include "ais/Dialect/AIS/IR/AISTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ais/Dialect/AIS/IR/AISDialect.h.inc"

#include "ais/Dialect/AIS/IR/AISOps.h"

#endif

/**
 * @file  AIS.td
 * @brief Top-level TableGen umbrella that pulls in every AIS sub-component.
 *
 * Include this file when you want to generate *all* AIS dialect artifacts
 * (dialect, types, attributes, operations) in one TableGen invocation.
 */

#ifndef APXM_AIS
#define APXM_AIS

include "ais/Dialect/AIS/IR/AISDialect.td"
include "ais/Dialect/AIS/IR/AISAttributes.td"
include "ais/Dialect/AIS/IR/AISTypes.td"
include "ais/Dialect/AIS/IR/AISOps.td"

#endif // APXM_AIS

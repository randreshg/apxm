/**
 * @file  AIS.td
 * @brief Top-level TableGen umbrella that pulls in every AIS sub-component.
 *
 * Include this file when you want to generate *all* AIS dialect artifacts
 * (dialect, types, attributes, operations) in one TableGen invocation.
 */

#ifndef APXM_AIS
#define APXM_AIS

include "apxm/Dialect/AIS/IR/AISDialect.td"
include "apxm/Dialect/AIS/IR/AISAttributes.td"
include "apxm/Dialect/AIS/IR/AISTypes.td"
include "apxm/Dialect/AIS/IR/AISOps.td"

#endif // APXM_AIS

#!/bin/bash
# NAME: timestamp
# DESCRIPTION: Display current timestamp in various formats
# ARG: format:string:Format type (iso, unix, human, all)

format="${1:-human}"

case "$format" in
    iso)
        date -u +"%Y-%m-%dT%H:%M:%SZ"
        ;;
    unix)
        date +%s
        ;;
    human)
        date "+%A, %B %d, %Y at %I:%M %p"
        ;;
    all)
        echo "ISO 8601: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "Unix timestamp: $(date +%s)"
        echo "Human readable: $(date "+%A, %B %d, %Y at %I:%M %p")"
        echo "RFC 2822: $(date -R)"
        ;;
    *)
        echo "Unknown format: $format"
        echo "Available formats: iso, unix, human, all"
        exit 1
        ;;
esac
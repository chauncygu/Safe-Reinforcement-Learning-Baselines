struct UppaalQueryFailedException <: Exception
    message::AbstractString
end

function extract_query_results(query_results::AbstractString)
    results = []
    open(query_results) do file
        for line in eachline(file)
            m_mean = match(r"mean=([\d.e-]+)", line)
            aborted = occursin(r"EXCEPTION: |is time-locked.|-- Aborted.", line)

            if aborted
                throw(UppaalQueryFailedException(line))
            end

            if m_mean === nothing
                continue
            end

            push!(results, m_mean[1])
        end
    end

    results
end
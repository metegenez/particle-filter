classdef sensor < handle
    %sensor ED cihazinin lokasyonu input olacak sekilde, toptan
    %tutulabilmesi ve okunabilirligi arttirmak icin obje tanimlandi.
    
    properties
        enlem
        boylam
    end
    
    methods
        function obj = sensor(enlem, boylam)
            obj.enlem = enlem;
            obj.boylam = boylam;
        end
    end
end


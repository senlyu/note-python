class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums3=[]
        s=0
        i=0
        j=0
        while i<=(len(nums1)-1) or j<=(len(nums2)-1):
            if i>(len(nums1)-1):
                nums3.append(nums2[j])
                j=j+1
                s=s+1
            elif j>(len(nums2)-1):
                nums3.append(nums1[i])
                i=i+1
                s=s+1
            elif nums1[i]>nums2[j]:
                nums3.append(nums2[j])
                s=s+1
                j=j+1
            else:
                nums3.append(nums1[i])
                s=s+1
                i=i+1
        if s%2 ==0:
            result=(nums3[s//2-1]+nums3[s//2])/2
        else:
            result=nums3[s//2]
        return result

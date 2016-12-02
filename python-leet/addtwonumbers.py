def addTwoNumbers(l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    carry=0
    n=root=listnode(0)
    while l1 or l2 or carry:
        if l1:
            carry+=l1.val
            l1=l1.next
        if l2:
            carry+=l2.val
            l2=l2.next
        carry, val=divmod(carry,10)
        n.next=listnode(val)
        n=n.next
    return root.next

